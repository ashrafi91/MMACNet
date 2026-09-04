"""MMAC-Net -- multi-modal, multi-label attention CNN for ICD-9 coding.

This module is the released implementation transcribed by Appendix A of the
manuscript.  It implements, in order:

* a learnable Word2Vec-initialised token embedding with dropout (Sec. 3.8);
* a stem 1-D convolution followed by ``conv_block_depth - 1`` residual
  depthwise-separable blocks, each with a Squeeze-and-Excitation channel gate
  (Eq. 1) inside a residual connection (Eq. 2 / Algorithm A1 ``CONVRESBLOCK``);
* per-label attention through the projection ``U`` (Sec. 3.8, Table 3);
* a late-fusion tabular branch over the aggregated structured features
  (Algorithm A1 ``TABULARENCODER``), gated per experiment by
  ``tabular_modalities`` so the Table 6 ablation rows are single-line configs;
* two ``tanh`` fully-connected refinement layers and a per-label linear head
  producing raw logits (no activation).

The optional description regulariser (``lmbda`` > 0) ties each label's attention
vector ``U[l]`` to a convolutional embedding of that ICD-9 code's textual
description (Sec. 3.9 item 4, Eq. 5, Algorithm A2).
"""

from math import floor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.init import xavier_uniform_

from MMACNet.utils.mapper import ConfigMapper
from MMACNet.utils.model_utils import load_lookups, pad_desc_vecs
from MMACNet.utils.text_loggers import get_logger

logger = get_logger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





CATEGORICAL_FIELDS = (
    "DRUG_TYPE",
    "DRUG",
    "PROD_STRENGTH",
    "ROUTE",
    "ORG_ITEMID",
    "AB_ITEMID",
    "INTERPRETATION",
)
NUMERICAL_FIELDS = ("DOSE_VAL_RX", "DILUTION_VALUE")


class BaseModel(nn.Module):
    """Shared embedding layer + lookup dictionaries."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.Y = config.num_classes
        self.embed_drop = nn.Dropout(p=config.dropout)

        self.dicts = load_lookups(
            dataset_dir=config.dataset_dir,
            mimic_dir=config.mimic_dir,
            static_dir=config.static_dir,
            word2vec_dir=config.word2vec_dir,
            version=config.version,
        )

        embedding_cls = ConfigMapper.get_object("embeddings", "word2vec")
        W = torch.Tensor(embedding_cls.load_emb_matrix(config.word2vec_dir))
        self.embed = nn.Embedding(W.size(0), W.size(1), padding_idx=0)
        self.embed.weight.data = W.clone()


    def embed_descriptions(self, desc_data):
        """Encode each label description with the description conv module."""
        param_device = next(self.parameters()).device
        b_batch = []
        for inst in desc_data:
            if len(inst) > 0:
                lt = Variable(torch.LongTensor(inst).to(param_device))
                d = self.desc_embedding(lt).transpose(1, 2)
                d = self.label_conv(d)
                d = F.max_pool1d(torch.tanh(d), kernel_size=d.size(2))
                d = d.squeeze(2)
                b_batch.append(self.label_fc1(d))
            else:
                b_batch.append([])
        return b_batch

    def _compare_label_embeddings(self, target, b_batch, desc_data):
        """L2 gap between the per-label attention vectors ``U[l]`` and the
        description embeddings ``b_l`` (Sec. 3.9 item 4)."""
        diffs = []
        for i, bi in enumerate(b_batch):
            if isinstance(bi, list):
                continue
            inds = torch.nonzero(target[i].data).squeeze().cpu().numpy()
            zi = self.U.weight[inds, :]
            diff = (zi - bi).mul(zi - bi).mean()
            diffs.append(self.config.lmbda * diff * bi.size(0))
        return diffs


class SEBlock1d(nn.Module):
    """Squeeze-and-Excitation channel gate, Eq. 1:
    ``SE(h) = sigmoid(W2 relu(W1 GAP(h))) (*) h``.
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        s = x.mean(dim=2)
        s = self.fc(s).unsqueeze(2)
        return x * s


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise (per-channel) conv followed by a pointwise 1x1 conv."""

    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            padding=padding, groups=in_channels, bias=False,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=True)
        xavier_uniform_(self.depthwise.weight)
        xavier_uniform_(self.pointwise.weight)

    @property
    def weight(self):
        return self.pointwise.weight

    @property
    def bias(self):
        return self.pointwise.bias

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class ConvResBlock(nn.Module):
    """Algorithm A1 ``CONVRESBLOCK``:

    ``Q = Conv1d(P); [BN]; Q = Dropout(relu(Q)); [SE]; [Q = Q + R]; return Q``

    where ``R = P`` when the channel counts match, else a 1x1 conv + BN
    projection of ``P``.  There is no activation after the residual add, which
    matches Algorithm A1 (Eq. 2 shows a post-add ReLU -- see
    docs/RECONCILIATION.md
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        use_batch_norm=True,
        activation="relu",
        dropout=0.0,
        use_residual=True,
        use_se=False,
        se_reduction=4,
        use_depthwise_separable=False,
    ):
        super().__init__()
        if use_depthwise_separable:
            self.conv = DepthwiseSeparableConv1d(
                in_channels, out_channels, kernel_size, padding=padding
            )
        else:
            self.conv = nn.Conv1d(
                in_channels, out_channels, kernel_size, padding=padding
            )
            xavier_uniform_(self.conv.weight)

        post = []
        if use_batch_norm:
            post.append(nn.BatchNorm1d(out_channels))
        act = self._make_activation(activation)
        if act is not None:
            post.append(act)
        if dropout and dropout > 0:
            post.append(nn.Dropout(p=dropout))
        self.post = nn.Sequential(*post) if post else nn.Identity()

        self.se = SEBlock1d(out_channels, se_reduction) if use_se else None

        self.use_residual = use_residual
        self.residual_proj = None
        if use_residual and in_channels != out_channels:
            self.residual_proj = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels),
            )
            xavier_uniform_(self.residual_proj[0].weight)

    @staticmethod
    def _make_activation(name):
        if not name:
            return None
        name = name.strip().lower() if isinstance(name, str) else name
        if name in (None, "none", "identity"):
            return None
        table = {
            "relu": nn.ReLU(inplace=True),
            "gelu": nn.GELU(),
            "selu": nn.SELU(inplace=True),
            "elu": nn.ELU(inplace=True),
            "leaky_relu": nn.LeakyReLU(inplace=True),
            "tanh": nn.Tanh(),
        }
        return table.get(name, nn.ReLU(inplace=True))

    def get_raw_conv(self):
        if isinstance(self.conv, DepthwiseSeparableConv1d):
            return self.conv.pointwise
        return self.conv

    def forward(self, x):
        identity = x
        out = self.post(self.conv(x))
        if self.se is not None:
            out = self.se(out)
        if self.use_residual:
            if self.residual_proj is not None:
                identity = self.residual_proj(identity)
            if identity.size(2) != out.size(2):
                diff = identity.size(2) - out.size(2)
                left = diff // 2
                identity = identity[:, :, left: left + out.size(2)]
            out = out + identity
        return out


@ConfigMapper.map("models", "MMACNet")
class ConvAttnPool(BaseModel):
    """The MMAC-Net architecture of Fig. 3 / Table 3."""

    def __init__(self, config):
        logger.info("Initializing MMAC-Net (ConvAttnPool)")
        super().__init__(config=config)

        self.pad_idx = self.dicts["w2ind"][config.pad_token]
        self.unk_idx = self.dicts["w2ind"][config.unk_token]


        conv_depth = max(1, getattr(config, "conv_block_depth", 6))
        reduced_kernel = max(1, config.kernel_size // 2)
        if reduced_kernel % 2 == 0:
            reduced_kernel += 1
        kernel_sizes = [config.kernel_size] + [
            reduced_kernel for _ in range(conv_depth - 1)
        ]

        self.use_batch_norm = getattr(config, "use_batch_norm", True)
        self.conv_activation = getattr(config, "conv_activation", "relu")
        self.conv_block_dropout = getattr(config, "conv_block_dropout", 0.0)
        self.use_residual = getattr(config, "use_residual", True)
        self.use_se_block = getattr(config, "use_se_block", True)
        self.se_reduction = getattr(config, "se_reduction", 4)
        self.use_depthwise_separable = getattr(
            config, "use_depthwise_separable", True
        )

        self.conv_blocks = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        in_channels = config.embed_size
        for block_idx, kernel_size in enumerate(kernel_sizes):


            use_dsc = self.use_depthwise_separable and block_idx > 0
            block = ConvResBlock(
                in_channels=in_channels,
                out_channels=config.num_filter_maps,
                kernel_size=kernel_size,
                padding=int(floor(kernel_size / 2)),
                use_batch_norm=self.use_batch_norm,
                activation=self.conv_activation,
                dropout=self.conv_block_dropout,
                use_residual=self.use_residual,
                use_se=self.use_se_block,
                se_reduction=self.se_reduction,
                use_depthwise_separable=use_dsc,
            )
            self.conv_blocks.append(block)
            self.conv_layers.append(block.get_raw_conv())
            in_channels = config.num_filter_maps
        self.conv = self.conv_layers[0]


        self.U = nn.Linear(config.num_filter_maps, self.Y)
        xavier_uniform_(self.U.weight)


        self.tabular_meta = self.dicts.get("tabular_meta")
        self.tabular_modalities = self._resolve_tabular_modalities(config)
        self.tabular_cat_order = []
        self.tabular_num_order = []
        self.tabular_cat_embeddings = nn.ModuleDict()
        self.tabular_numeric_bn = None
        self.tabular_mlp = None
        tabular_extra_dim = self._build_tabular_branch()

        base_feature_dim = config.num_filter_maps + (tabular_extra_dim or 0)


        self.fc_dropout_rate = getattr(config, "fc_dropout", config.dropout)
        self.fc_dropout_layer = (
            nn.Dropout(p=self.fc_dropout_rate)
            if self.fc_dropout_rate and self.fc_dropout_rate > 0
            else None
        )
        fc_layer_dims = list(
            getattr(config, "fc_layer_dims", [config.num_filter_maps])
        )
        self.fc_layers = nn.ModuleList()
        prev_dim = base_feature_dim
        for dim in fc_layer_dims:
            layer = nn.Linear(prev_dim, dim)
            xavier_uniform_(layer.weight)
            self.fc_layers.append(layer)
            prev_dim = dim
        feature_dim_after_fc = prev_dim

        self.final = nn.Linear(feature_dim_after_fc, self.Y)
        xavier_uniform_(self.final.weight)


        if getattr(config, "lmbda", 0.0) and config.lmbda > 0:
            W = self.embed.weight.data
            self.desc_embedding = nn.Embedding(
                W.size(0), W.size(1), padding_idx=0
            )
            self.desc_embedding.weight.data = W.clone()
            self.label_conv = nn.Conv1d(
                config.embed_size,
                config.num_filter_maps,
                kernel_size=config.kernel_size,
                padding=int(floor(config.kernel_size / 2)),
            )
            xavier_uniform_(self.label_conv.weight)
            self.label_fc1 = nn.Linear(
                config.num_filter_maps, config.num_filter_maps
            )
            xavier_uniform_(self.label_fc1.weight)

            self.dv_dict = {}
            ind2c = self.dicts["ind2c"]
            w2ind = self.dicts["w2ind"]
            desc_dict = self.dicts["desc"]
            for i, c in ind2c.items():
                desc = desc_dict.get(c, "") if hasattr(desc_dict, "get") else desc_dict[c]
                self.dv_dict[i] = [
                    w2ind.get(w, self.unk_idx) for w in str(desc).split()
                ]


    @staticmethod
    def _resolve_tabular_modalities(config):
        """Return the set of enabled structured modalities.

        ``None``  -> use whatever ``tabular_meta.json`` contains (both groups);
        ``[]``    -> text-only (Table 6 "Notes (Baseline)");
        list      -> restrict to the named groups ("categorical" / "numerical").
        """
        modalities = getattr(config, "tabular_modalities", None)
        if modalities is None:
            return None
        return {str(m).strip().lower() for m in modalities}

    def _modality_enabled(self, group):
        if self.tabular_modalities is None:
            return True
        return group in self.tabular_modalities


    def _build_tabular_branch(self):
        if not self.tabular_meta:
            return 0
        cat_order = self.tabular_meta.get("categorical_order", [])
        num_order = self.tabular_meta.get("numerical_order", [])
        if self.tabular_modalities is not None and not self.tabular_modalities:
            return 0
        if not self._modality_enabled("categorical"):
            cat_order = []
        if not self._modality_enabled("numerical"):
            num_order = []
        if not cat_order and not num_order:
            return 0

        self.tabular_cat_order = cat_order
        self.tabular_num_order = num_order

        cat_total_dim = 0
        for col in cat_order:
            num_classes = self.tabular_meta["categorical"][col]["num_classes"]
            emb_dim = min(64, max(4, num_classes // 4))
            emb = nn.Embedding(num_classes, emb_dim, padding_idx=0)
            xavier_uniform_(emb.weight)
            self.tabular_cat_embeddings[col] = emb
            cat_total_dim += emb_dim

        num_dim = len(num_order)
        if num_dim > 0:
            self.tabular_numeric_bn = nn.BatchNorm1d(num_dim)

        combined_dim = cat_total_dim + num_dim
        if combined_dim == 0:
            return 0

        tab_hidden = getattr(self.config, "tabular_hidden_dim", 50)
        tab_dropout = getattr(
            self.config, "tabular_dropout", self.config.dropout
        )
        self.tabular_mlp = nn.Sequential(
            nn.Linear(combined_dim, tab_hidden),
            nn.ReLU(),
            nn.Dropout(p=tab_dropout),
        )
        return tab_hidden

    def _forward_tabular_branch(self, categorical=None, numerical=None):
        if self.tabular_mlp is None:
            return None
        features = []
        if self.tabular_cat_order:
            if categorical is None:
                raise ValueError("Categorical inputs required by the tabular branch.")
            embs = [
                self.tabular_cat_embeddings[col](categorical[:, idx])
                for idx, col in enumerate(self.tabular_cat_order)
            ]
            features.append(torch.cat(embs, dim=1))
        if self.tabular_num_order:
            if numerical is None:
                raise ValueError("Numerical inputs required by the tabular branch.")
            num_repr = numerical
            if self.tabular_numeric_bn is not None:
                num_repr = self.tabular_numeric_bn(num_repr)
            features.append(num_repr)
        if not features:
            return None
        return self.tabular_mlp(torch.cat(features, dim=1))


    def forward(self, text, categorical=None, numerical=None):
        x = self.embed(text)
        x = self.embed_drop(x)
        x = x.transpose(1, 2)

        for block in self.conv_blocks:
            x = block(x)
        x = x.transpose(1, 2)


        self.alpha = F.softmax(self.U.weight.matmul(x.transpose(1, 2)), dim=2)
        m = self.alpha.matmul(x)

        tabular_repr = self._forward_tabular_branch(categorical, numerical)
        if tabular_repr is not None:
            m = torch.cat(
                [m, tabular_repr.unsqueeze(1).expand(-1, m.size(1), -1)], dim=2
            )

        if self.fc_layers:
            b, y, feat = m.size()
            doc = m.view(-1, feat)
            for layer in self.fc_layers:
                doc = torch.tanh(layer(doc))
                if self.fc_dropout_layer is not None:
                    doc = self.fc_dropout_layer(doc)
            m = doc.view(b, y, -1)

        return self.final.weight.mul(m).sum(dim=2).add(self.final.bias)


    def regularizer(self, labels=None):
        if not getattr(self.config, "lmbda", 0.0):
            return 0.0
        desc_vecs = [
            [self.dv_dict[i] for i, on in enumerate(label) if on and self.dv_dict[i]]
            for label in labels
        ]
        desc_data = [np.array(pad_desc_vecs(dvs)) if dvs else np.zeros((0, 0)) for dvs in desc_vecs]
        b_batch = self.embed_descriptions(desc_data)
        diffs = self._compare_label_embeddings(labels, b_batch, desc_data)
        if not diffs:
            return torch.tensor(0.0, device=self.final.weight.device)
        return torch.stack(diffs).mean()

    def get_input_attention(self):
        """Per-label token attention distribution (Sec. 7.4 explainability)."""
        return self.alpha.cpu().detach().numpy()

    def predict_one_hot(self, text, categorical=None, numerical=None):
        logits = self.forward(text, categorical, numerical)
        idx = torch.argmax(logits, dim=1)
        return F.one_hot(idx, num_classes=self.Y).float()


@ConfigMapper.map("models", "CNN")
class VanillaConv(BaseModel):
    """Single-conv text-only CNN (the CAML-lineage baseline of Table 7)."""

    def __init__(self, config):
        logger.info("Initializing CNN (VanillaConv)")
        super().__init__(config)
        self.conv = nn.Conv1d(
            config.embed_size, config.num_filter_maps, kernel_size=config.kernel_size
        )
        xavier_uniform_(self.conv.weight)
        self.fc = nn.Linear(config.num_filter_maps, self.Y)
        xavier_uniform_(self.fc.weight)

    def forward(self, text, **_):
        x = self.embed(text)
        x = self.embed_drop(x)
        x = x.transpose(1, 2)
        c = self.conv(x)
        x = F.max_pool1d(torch.tanh(c), kernel_size=c.size(2)).squeeze(dim=2)
        return self.fc(x)

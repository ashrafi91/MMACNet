
import math
from math import floor
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.init import xavier_uniform

from MMACNet.utils.mapper import ConfigMapper
from MMACNet.utils.model_utils import load_lookups, pad_desc_vecs
from MMACNet.utils.text_loggers import get_logger

logger = get_logger(__name__)

# Prefer CUDA; fall back to CPU if unavailable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# From learn/models.py
class BaseModel(nn.Module):
    def __init__(self, config):
        super(BaseModel, self).__init__()
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

        # make embedding layer
        embedding_cls = ConfigMapper.get_object("embeddings", "word2vec")
        W = torch.Tensor(embedding_cls.load_emb_matrix(config.word2vec_dir))
        self.embed = nn.Embedding(W.size()[0], W.size()[1], padding_idx=0)
        self.embed.weight.data = W.clone()

    def embed_descriptions(self, desc_data):
        # label description embedding via convolutional layer
        # number of labels is inconsistent across instances, so have to iterate
        # over the batch

        # Determine the device from model parameters
        param_device = next(self.parameters()).device

        b_batch = []
        for inst in desc_data:
            if len(inst) > 0:
                lt = Variable(torch.LongTensor(inst).to(param_device))
                d = self.desc_embedding(lt)
                d = d.transpose(1, 2)
                d = self.label_conv(d)
                d = F.max_pool1d(F.tanh(d), kernel_size=d.size()[2])
                d = d.squeeze(2)
                b_inst = self.label_fc1(d)
                b_batch.append(b_inst)
            else:
                b_batch.append([])
        return b_batch

    def _compare_label_embeddings(self, target, b_batch, desc_data):
        # description regularization loss
        # b is the embedding from description conv
        # iterate over batch because each instance has different # labels
        diffs = []
        for i, bi in enumerate(b_batch):
            ti = target[i]
            inds = torch.nonzero(ti.data).squeeze().cpu().numpy()

            zi = self.final.weight[inds, :]
            diff = (zi - bi).mul(zi - bi).mean()

            # multiply by number of labels to make sure overall mean is balanced
            # with regard to number of labels
            diffs.append(self.config.lmbda * diff * bi.size()[0])
        return diffs


class GraphConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim, activation=None, dropout=0.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        xavier_uniform(self.linear.weight)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

    def forward(self, x, adj):
        h = torch.matmul(adj, self.linear(x))
        if self.dropout:
            h = self.dropout(h)
        if self.activation:
            h = self.activation(h)
        return h


class DirGraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, activation=None, dropout=0.0):
        super().__init__()
        self.w_in = nn.Linear(in_dim, out_dim, bias=False)
        self.w_out = nn.Linear(in_dim, out_dim, bias=False)
        self.w_self = nn.Linear(in_dim, out_dim, bias=False)
        xavier_uniform(self.w_in.weight)
        xavier_uniform(self.w_out.weight)
        xavier_uniform(self.w_self.weight)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

    def forward(self, x, adj_in, adj_out):
        h_in = torch.matmul(adj_in, self.w_in(x))
        h_out = torch.matmul(adj_out, self.w_out(x))
        h_self = self.w_self(x)
        h = h_in + h_out + h_self
        if self.dropout:
            h = self.dropout(h)
        if self.activation:
            h = self.activation(h)
        return h


class GPSLabelBlock(nn.Module):
    def __init__(
        self,
        dim,
        heads=4,
        dropout=0.1,
        activation=F.relu,
    ):
        super().__init__()
        self.local = GraphConvLayer(dim, dim, activation=activation, dropout=dropout)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, adj):
        local = self.local(h, adj)
        h = self.norm1(h + self.dropout(local))
        attn_out, _ = self.attn(h, h, h)
        h = self.norm2(h + self.dropout(attn_out))
        return h


class DepthwiseSeparableConv1d(nn.Module):
    """Factored 1-D convolution: depthwise (per-channel) + pointwise (1x1).

    Reduces parameters from ``in_channels * out_channels * kernel_size`` to
    ``in_channels * kernel_size + in_channels * out_channels``.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            padding=padding, groups=in_channels, bias=False,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=True)
        xavier_uniform(self.depthwise.weight)
        xavier_uniform(self.pointwise.weight)

    @property
    def weight(self):
        """Expose pointwise weight for backward-compat access."""
        return self.pointwise.weight

    @property
    def bias(self):
        """Expose pointwise bias for backward-compat access."""
        return self.pointwise.bias

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class SEBlock1d(nn.Module):
    """Squeeze-and-Excitation channel attention for 1-D feature maps."""

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
        # x: [B, C, T]
        s = x.mean(dim=2)              # global avg pool -> [B, C]
        s = self.fc(s).unsqueeze(2)     # -> [B, C, 1]
        return x * s


class ConvResBlock(nn.Module):
    """Conv1d block with optional residual skip, SE attention, and DSC.

    Supports:
    - Standard or depthwise-separable convolution
    - BatchNorm + configurable activation
    - Squeeze-and-Excitation channel attention
    - Residual skip connection (with 1x1 projection when dims differ)
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
        # Main convolution
        if use_depthwise_separable:
            self.conv = DepthwiseSeparableConv1d(
                in_channels, out_channels, kernel_size, padding=padding,
            )
        else:
            self.conv = nn.Conv1d(
                in_channels, out_channels, kernel_size, padding=padding,
            )
            xavier_uniform(self.conv.weight)

        # Post-conv: BN -> activation -> dropout
        layers = []
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(out_channels))
        act = self._make_activation(activation)
        if act is not None:
            layers.append(act)
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        self.post = nn.Sequential(*layers) if layers else nn.Identity()

        # Squeeze-and-Excitation
        self.se = SEBlock1d(out_channels, se_reduction) if use_se else None

        # Residual path
        self.use_residual = use_residual
        self.residual_proj = None
        if use_residual and in_channels != out_channels:
            self.residual_proj = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels),
            )
            xavier_uniform(self.residual_proj[0].weight)

    @staticmethod
    def _make_activation(name):
        if not name:
            return None
        name = name.strip().lower() if isinstance(name, str) else name
        if name in (None, "none", "identity"):
            return None
        activations = {
            "relu": nn.ReLU(inplace=True),
            "gelu": nn.GELU(),
            "selu": nn.SELU(inplace=True),
            "elu": nn.ELU(inplace=True),
            "leaky_relu": nn.LeakyReLU(inplace=True),
            "tanh": nn.Tanh(),
        }
        return activations.get(name, nn.ReLU(inplace=True))

    def get_raw_conv(self):
        """Return the underlying nn.Conv1d (for init_code_emb compat)."""
        if isinstance(self.conv, DepthwiseSeparableConv1d):
            return self.conv.pointwise
        return self.conv

    def forward(self, x):
        identity = x
        out = self.conv(x)
        out = self.post(out)
        if self.se is not None:
            out = self.se(out)
        if self.use_residual:
            if self.residual_proj is not None:
                identity = self.residual_proj(identity)
            # Safety: handle potential sequence length mismatch
            if identity.size(2) != out.size(2):
                diff = identity.size(2) - out.size(2)
                left = diff // 2
                identity = identity[:, :, left: left + out.size(2)]
            out = out + identity
        return out


@ConfigMapper.map("models", "MMACNet")
class ConvAttnPool(BaseModel):
    def __init__(self, config):
        cls_name = self.__class__.__name__
        logger.info(f"Initializing {cls_name}")
        logger.debug(f"Initializing {cls_name} with config: {config}")
        super(ConvAttnPool, self).__init__(config=config)

        self.pad_idx = self.dicts["w2ind"][config.pad_token]
        self.unk_idx = self.dicts["w2ind"][config.unk_token]

        # initialize conv stack as in 2.1 but allow deeper blocks
        conv_kernel_sizes = getattr(config, "conv_kernel_sizes", None)
        conv_depth = max(1, getattr(config, "conv_block_depth", 2))
        if not conv_kernel_sizes:
            if conv_depth == 1:
                conv_kernel_sizes = [config.kernel_size]
            else:
                reduced_kernel = max(1, config.kernel_size // 2)
                if reduced_kernel % 2 == 0:
                    reduced_kernel += 1
                conv_kernel_sizes = [config.kernel_size] + [
                    reduced_kernel for _ in range(conv_depth - 1)
                ]

        self.use_batch_norm = getattr(config, "use_batch_norm", True)
        self.conv_activation = getattr(config, "conv_activation", "relu")
        self.conv_block_dropout = getattr(config, "conv_block_dropout", 0.1)
        self.use_residual = getattr(config, "use_residual", False)
        self.use_se_block = getattr(config, "use_se_block", False)
        self.se_reduction = getattr(config, "se_reduction", 4)
        self.use_depthwise_separable = getattr(
            config, "use_depthwise_separable", False
        )
        self.conv_layers = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()
        in_channels = config.embed_size
        for kernel_idx, kernel_size in enumerate(conv_kernel_sizes):
            # First block always uses standard Conv1d (init_code_emb compat)
            use_dsc = self.use_depthwise_separable and kernel_idx > 0
            block = ConvResBlock(
                in_channels=in_channels,
                out_channels=config.num_filter_maps,
                kernel_size=kernel_size,
                padding=int(floor(kernel_size / 2)),
                use_batch_norm=self.use_batch_norm,
                activation=self.conv_activation,
                dropout=self.conv_block_dropout if self.conv_block_dropout else 0.0,
                use_residual=self.use_residual,
                use_se=self.use_se_block,
                se_reduction=self.se_reduction,
                use_depthwise_separable=use_dsc,
            )
            self.conv_blocks.append(block)
            self.conv_layers.append(block.get_raw_conv())
            in_channels = config.num_filter_maps
        # Maintain reference for backward compatibility with any external hooks
        self.conv = self.conv_layers[0]

        # context vectors for computing attention as in 2.2
        self.U = nn.Linear(config.num_filter_maps, self.Y)
        xavier_uniform(self.U.weight)

        self.tabular_meta = self.dicts.get("tabular_meta")
        self.tabular_cat_order = []
        self.tabular_num_order = []
        self.tabular_cat_embeddings = nn.ModuleDict()
        self.tabular_numeric_bn = None
        self.tabular_mlp = None
        tabular_extra_dim = self._build_tabular_branch()

        base_feature_dim = config.num_filter_maps + (tabular_extra_dim or 0)

        self.fc_layers = nn.ModuleList()
        self.fc_dropout_rate = getattr(
            config, "fc_dropout", getattr(config, "dropout", 0.2)
        )
        self.fc_dropout_layer = (
            nn.Dropout(p=self.fc_dropout_rate)
            if self.fc_dropout_rate and self.fc_dropout_rate > 0
            else None
        )
        fc_layer_dims = getattr(config, "fc_layer_dims", [config.num_filter_maps])
        prev_dim = base_feature_dim
        for dim in fc_layer_dims:
            layer = nn.Linear(prev_dim, dim)
            xavier_uniform(layer.weight)
            self.fc_layers.append(layer)
            prev_dim = dim
        feature_dim_after_fc = prev_dim if self.fc_layers else base_feature_dim

        # final layer: create a matrix to use for the L binary classifiers as in
        # 2.3
        final_in_dim = feature_dim_after_fc
        self.final = nn.Linear(final_in_dim, self.Y)
        xavier_uniform(self.final.weight)

        # initialize with trained code embeddings if applicable
        if config.init_code_emb:
            self._code_emb_init()

            if config.embed_size != config.num_filter_maps:
                logger.warning(
                    "Cannot init convolution weights since the dimension differ "
                    "from the dimension of the embedding"
                )
            else:
                # also set first conv weights to do sum of inputs
                weights = (
                    torch.eye(config.embed_size)
                    .unsqueeze(2)
                    .expand(-1, -1, config.kernel_size)
                    / config.kernel_size
                )
                first_conv = self.conv_layers[0]
                first_conv.weight.data = weights.clone()
                first_conv.bias.data.zero_()

        # conv for label descriptions as in 2.5
        # description module has its own embedding and convolution layers
        if config.lmbda > 0:
            W = self.embed.weight.data
            self.desc_embedding = nn.Embedding(
                W.size()[0], W.size()[1], padding_idx=0
            )
            self.desc_embedding.weight.data = W.clone()

            self.label_conv = nn.Conv1d(
                config.embed_size,
                config.num_filter_maps,
                kernel_size=config.kernel_size,
                padding=int(floor(config.kernel_size / 2)),
            )
            xavier_uniform(self.label_conv.weight)

            self.label_fc1 = nn.Linear(
                config.num_filter_maps, config.num_filter_maps
            )
            xavier_uniform(self.label_fc1.weight)

            # Pre-process the code description into word idxs
            self.dv_dict = {}
            ind2c = self.dicts["ind2c"]
            w2ind = self.dicts["w2ind"]
            desc_dict = self.dicts["desc"]
            for i, c in ind2c.items():
                desc_vec = [
                    w2ind[w] if w in w2ind else self.unk_idx
                    for w in desc_dict[c]
                ]
                self.dv_dict[i] = desc_vec

    def _code_emb_init(self):
        # In the original CAML repo, this method seems not being called.
        # In this implementation, we compute the AVERAGE word2vec embeddings for
        # each code and initialize the self.U and self.final with it.
        ind2c = self.dicts["ind2c"]
        w2ind = self.dicts["w2ind"]
        desc_dict = self.dicts["desc"]

        embed_dim = self.embed.embedding_dim
        param_device = self.final.weight.device
        weights = torch.zeros(
            self.Y,
            embed_dim,
            device=param_device,
            dtype=self.final.weight.dtype,
        )
        embed_device = self.embed.weight.device
        for i, c in ind2c.items():
            desc_vec = [
                w2ind[w] if w in w2ind else self.unk_idx
                for w in desc_dict[c].split()
            ]
            if not desc_vec:
                continue
            token_tensor = torch.tensor(
                desc_vec, device=embed_device, dtype=torch.long
            )
            weights[i] = self.embed(token_tensor).mean(dim=0).to(weights.device)

        if self.U.in_features == embed_dim:
            self.U.weight.data = weights.clone()
        else:
            logger.warning(
                "Cannot init attention vectors since their dimension differs "
                "from the embedding dimension"
            )

        if self.final.in_features == embed_dim:
            self.final.weight.data = weights.clone()
        else:
            logger.warning(
                "Cannot init final layer since its input dimension differs "
                "from the embedding dimension"
            )

    def forward(self, text, categorical=None, numerical=None):
        # get embeddings and apply dropout
        x = self.embed(text)
        x = self.embed_drop(x)
        x = x.transpose(1, 2)

        # apply convolutional stack with optional batch normalization
        for block in self.conv_blocks:
            x = block(x)
        x = x.transpose(1, 2)
        # apply attention
        self.alpha = F.softmax(self.U.weight.matmul(x.transpose(1, 2)), dim=2)
        # document representations are weighted sums using the attention. Can
        # compute all at once as a matmul
        m = self.alpha.matmul(x)
        tabular_repr = self._forward_tabular_branch(
            categorical_inputs=categorical, numerical_inputs=numerical
        )
        if tabular_repr is not None:
            tabular_expanded = tabular_repr.unsqueeze(1).expand(
                -1, m.size(1), -1
            )
            m = torch.cat([m, tabular_expanded], dim=2)
        # optional fully connected refinement before classification
        if self.fc_layers:
            batch_size, num_labels, feat_dim = m.size()
            doc_repr = m.view(-1, feat_dim)
            for layer in self.fc_layers:
                doc_repr = torch.tanh(layer(doc_repr))
                if self.fc_dropout_layer is not None:
                    doc_repr = self.fc_dropout_layer(doc_repr)
            m = doc_repr.view(batch_size, num_labels, -1)
        # final layer classification
        y = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)

        return y

    def predict_one_hot(self, text, categorical=None, numerical=None):
        logits = self.forward(text, categorical, numerical)
        idx = torch.argmax(logits, dim=1)
        return F.one_hot(idx, num_classes=self.Y).float()


    def regularizer(self, labels=None):
        if not self.config.lmbda:
            return 0.0

        # Retrive the description tokens of the labels
        desc_vecs = []
        for label in labels:
            desc_vecs.append(
                [self.dv_dict[i] for i, l in enumerate(label) if l]
            )
        desc_data = [np.array(pad_desc_vecs(dvs)) for dvs in desc_vecs]

        # run descriptions through description module
        b_batch = self.embed_descriptions(desc_data)
        # get l2 similarity loss
        diffs = self._compare_label_embeddings(labels, b_batch, desc_data)
        diff = torch.stack(diffs).mean()

        return diff

    def get_input_attention(self):
        # Use the attention score computed in the forward pass
        return self.alpha[:, :, :-1].cpu().detach().numpy()

    def _build_conv_activation_module(self):
        if not self.conv_activation:
            return None
        name = (
            self.conv_activation.strip().lower()
            if isinstance(self.conv_activation, str)
            else self.conv_activation
        )
        if name in {None, "none", "identity"}:
            return None
        if name == "relu":
            return nn.ReLU()
        if name == "gelu":
            return nn.GELU()
        if name == "selu":
            return nn.SELU()
        if name == "elu":
            return nn.ELU()
        if name == "leaky_relu":
            return nn.LeakyReLU()
        if name == "tanh":
            return nn.Tanh()
        logger.warning(
            f"Unknown conv activation {self.conv_activation}, defaulting to ReLU."
        )
        return nn.ReLU()

    def _build_tabular_branch(self):
        if not self.tabular_meta:
            return 0
        categorical_order = self.tabular_meta.get("categorical_order", [])
        numerical_order = self.tabular_meta.get("numerical_order", [])
        if not categorical_order and not numerical_order:
            return 0

        self.tabular_cat_order = categorical_order
        self.tabular_num_order = numerical_order

        cat_total_dim = 0
        for col in self.tabular_cat_order:
            meta = self.tabular_meta["categorical"][col]
            num_classes = meta["num_classes"]
            emb_dim = min(64, max(4, num_classes // 4))
            self.tabular_cat_embeddings[col] = nn.Embedding(
                num_classes, emb_dim, padding_idx=0
            )
            xavier_uniform(self.tabular_cat_embeddings[col].weight)
            cat_total_dim += emb_dim

        num_dim = len(self.tabular_num_order)
        if num_dim > 0:
            self.tabular_numeric_bn = nn.BatchNorm1d(num_dim)

        combined_dim = cat_total_dim + num_dim
        if combined_dim == 0:
            return 0

        tab_hidden = getattr(
            self.config,
            "tabular_hidden_dim",
            max(32, min(128, combined_dim * 2)),
        )
        tab_dropout = getattr(
            self.config,
            "tabular_dropout",
            getattr(self.config, "dropout", 0.2),
        )
        self.tabular_mlp = nn.Sequential(
            nn.Linear(combined_dim, tab_hidden),
            nn.ReLU(),
            nn.Dropout(p=tab_dropout),
        )

        return tab_hidden

    def _forward_tabular_branch(self, categorical_inputs=None, numerical_inputs=None):
        if self.tabular_mlp is None:
            return None

        features = []
        if self.tabular_cat_order:
            if categorical_inputs is None:
                raise ValueError(
                    "Categorical inputs missing while tabular embeddings are enabled."
                )
            cat_embeddings = []
            for idx, col in enumerate(self.tabular_cat_order):
                cat_embeddings.append(
                    self.tabular_cat_embeddings[col](categorical_inputs[:, idx])
                )
            features.append(torch.cat(cat_embeddings, dim=1))

        if self.tabular_num_order:
            if numerical_inputs is None:
                raise ValueError(
                    "Numerical inputs missing while tabular normalization is enabled."
                )
            num_repr = numerical_inputs
            if self.tabular_numeric_bn is not None:
                num_repr = self.tabular_numeric_bn(num_repr)
            features.append(num_repr)

        if not features:
            return None

        concat = torch.cat(features, dim=1)
        return self.tabular_mlp(concat)


@ConfigMapper.map("models", "CNN")
class VanillaConv(BaseModel):
    def __init__(self, config):
        cls_name = self.__class__.__name__
        logger.info(f"Initializing {cls_name}")
        logger.debug(f"Initializing {cls_name} with config: {config}")
        super(VanillaConv, self).__init__(config)

        # initialize conv layer as in 2.1
        self.conv = nn.Conv1d(
            config.embed_size,
            config.num_filter_maps,
            kernel_size=config.kernel_size,
        )
        xavier_uniform(self.conv.weight)

        # linear output
        self.fc = nn.Linear(config.num_filter_maps, self.Y)
        xavier_uniform(self.fc.weight)

    def forward(self, text, **_):
        # embed
        x = self.embed(text)
        x = self.embed_drop(x)
        x = x.transpose(1, 2)

        # conv/max-pooling
        c = self.conv(x)
        x = F.max_pool1d(F.tanh(c), kernel_size=c.size()[2])
        x = x.squeeze(dim=2)

        # linear output
        x = self.fc(x)

        return x


@ConfigMapper.map("models", "MMACNetGNN")
class GraphConvAttnPool(BaseModel):
    def __init__(self, config):
        cls_name = self.__class__.__name__
        logger.info(f"Initializing {cls_name}")
        logger.debug(f"Initializing {cls_name} with config: {config}")
        super(GraphConvAttnPool, self).__init__(config=config)

        self.pad_idx = self.dicts["w2ind"][config.pad_token]
        self.unk_idx = self.dicts["w2ind"][config.unk_token]

        self.conv = nn.Conv1d(
            config.embed_size,
            config.num_filter_maps,
            kernel_size=config.kernel_size,
            padding=int(floor(config.kernel_size / 2)),
        )
        xavier_uniform(self.conv.weight)

        self.graph_feature_dim = getattr(
            config, "label_feature_dim", config.num_filter_maps
        )
        self.label_features = nn.Parameter(
            torch.empty(self.Y, self.graph_feature_dim)
        )
        xavier_uniform(self.label_features)

        self.label_encoder_type = getattr(
            config, "label_encoder_type", "gcn"
        ).lower()
        adj = self._load_label_graph()
        if self.label_encoder_type == "dir":
            adj_in, adj_out = self._prepare_directional_adj(adj)
            self.register_buffer("label_adj_in", adj_in)
            self.register_buffer("label_adj_out", adj_out)
        else:
            self.register_buffer("label_adj", adj)

        hidden_dims = getattr(config, "label_graph_hidden_dims", [])
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        graph_dropout = getattr(config, "label_graph_dropout", 0.0)
        activation_spec = getattr(config, "label_graph_activation", "relu")
        activations = self._expand_activations(activation_spec, len(hidden_dims))

        self.label_edge_dropout = getattr(config, "label_edge_dropout", 0.0)
        self.label_stochastic_depth = getattr(
            config, "label_stochastic_depth", 0.0
        )
        self.label_global_heads = getattr(config, "label_global_heads", 2)

        self.graph_layers = nn.ModuleList()
        in_dim = self.graph_feature_dim
        if self.label_encoder_type == "gps":
            dims = hidden_dims or [in_dim]
            for layer_idx, hidden_dim in enumerate(dims):
                activation = activations[layer_idx] if activations else None
                heads = self._select_attn_heads(hidden_dim, self.label_global_heads)
                proj = None
                if hidden_dim != in_dim:
                    proj = nn.Linear(in_dim, hidden_dim, bias=False)
                    xavier_uniform(proj.weight)
                block = GPSLabelBlock(
                    dim=hidden_dim,
                    heads=heads,
                    dropout=graph_dropout,
                    activation=activation,
                )
                self.graph_layers.append(
                    nn.ModuleDict({"proj": proj, "block": block})
                )
                in_dim = hidden_dim
        else:
            for layer_idx, hidden_dim in enumerate(hidden_dims):
                activation = activations[layer_idx] if activations else None
                if self.label_encoder_type == "dir":
                    layer = DirGraphConv(
                        in_dim,
                        hidden_dim,
                        activation=activation,
                        dropout=graph_dropout,
                    )
                else:
                    layer = GraphConvLayer(
                        in_dim,
                        hidden_dim,
                        activation=activation,
                        dropout=graph_dropout,
                    )
                self.graph_layers.append(layer)
                in_dim = hidden_dim

        self.attn_proj = nn.Linear(in_dim, config.num_filter_maps, bias=False)
        xavier_uniform(self.attn_proj.weight)
        self.cls_proj = nn.Linear(in_dim, config.num_filter_maps, bias=False)
        xavier_uniform(self.cls_proj.weight)
        self.classifier_bias = nn.Parameter(torch.zeros(self.Y))

        self.use_text_summary_attn = getattr(
            config, "label_query_use_text_summary", True
        )
        if self.use_text_summary_attn:
            self.text_summary_proj = nn.Linear(
                2 * config.num_filter_maps, self.attn_proj.out_features, bias=False
            )
            xavier_uniform(self.text_summary_proj.weight)

        self._cached_label_states = None
        self._cached_classifier_weights = None
        self.alpha = None

        if config.lmbda > 0:
            W = self.embed.weight.data
            self.desc_embedding = nn.Embedding(
                W.size()[0], W.size()[1], padding_idx=0
            )
            self.desc_embedding.weight.data = W.clone()

            self.label_conv = nn.Conv1d(
                config.embed_size,
                config.num_filter_maps,
                kernel_size=config.kernel_size,
                padding=int(floor(config.kernel_size / 2)),
            )
            xavier_uniform(self.label_conv.weight)

            self.label_fc1 = nn.Linear(
                config.num_filter_maps, config.num_filter_maps
            )
            xavier_uniform(self.label_fc1.weight)

            self.dv_dict = {}
            ind2c = self.dicts["ind2c"]
            w2ind = self.dicts["w2ind"]
            desc_dict = self.dicts["desc"]
            for i, c in ind2c.items():
                desc_vec = [
                    w2ind[w] if w in w2ind else self.unk_idx
                    for w in desc_dict[c]
                ]
                self.dv_dict[i] = desc_vec

    def _expand_activations(self, activation_spec, length):
        if not length:
            return []

        if isinstance(activation_spec, (list, tuple)):
            activations = [
                self._resolve_activation_fn(name) for name in activation_spec
            ]
        else:
            activations = [self._resolve_activation_fn(activation_spec)]

        if not activations:
            activations = [None]

        if len(activations) < length:
            activations.extend([activations[-1]] * (length - len(activations)))

        return activations[:length]

    def _resolve_activation_fn(self, activation_name):
        if not activation_name:
            return None

        name = activation_name.lower()
        if name == "relu":
            return F.relu
        if name == "gelu":
            return F.gelu
        if name == "tanh":
            return torch.tanh
        if name in {"identity", "linear", "none"}:
            return None
        logger.warning(
            f"Unknown activation {activation_name}, defaulting to relu"
        )
        return F.relu

    def _load_label_graph(self):
        graph_path = getattr(self.config, "label_graph_path", None)
        adjacency = None
        if graph_path:
            expanded_path = os.path.expanduser(graph_path)
            if os.path.exists(expanded_path):
                adjacency = torch.from_numpy(np.load(expanded_path)).float()
                logger.info(f"Loaded label graph from {expanded_path}")
            else:
                logger.warning(
                    f"Label graph path {expanded_path} not found. "
                    "Falling back to heuristic construction."
                )

        use_desc_graph = getattr(
            self.config, "label_graph_use_desc_embeddings", False
        )
        if adjacency is None and use_desc_graph:
            top_k = getattr(self.config, "label_graph_top_k", 10)
            threshold = getattr(
                self.config, "label_graph_similarity_threshold", 0.0
            )
            adjacency = self._build_desc_similarity_graph(top_k, threshold)

        if adjacency is None:
            adjacency = torch.eye(self.Y)
            logger.info("Label graph defaulting to identity adjacency.")

        if adjacency.dim() == 1:
            adjacency = torch.diag(adjacency)

        if (
            adjacency.size(0) != self.Y
            or adjacency.size(1) != self.Y
        ):
            raise ValueError(
                "Label adjacency must be square with size equal to num_classes."
            )

        return self._normalize_adjacency(adjacency.float())

    def _prepare_directional_adj(self, adjacency):
        # Treat provided adjacency as possibly asymmetric; build in/out without
        # re-adding self loops to avoid double counting after normalization.
        adj_in = adjacency
        adj_out = adjacency.t()
        return adj_in, adj_out

    def _select_attn_heads(self, dim, preferred_heads):
        if dim % preferred_heads == 0:
            return preferred_heads
        for h in range(preferred_heads - 1, 0, -1):
            if dim % h == 0:
                logger.warning(
                    f"Adjusting label_global_heads from {preferred_heads} to {h} to divide dim {dim}."
                )
                return h
        logger.warning(
            f"Falling back to 1 attention head since dim {dim} is prime relative to requested heads {preferred_heads}."
        )
        return 1

    def _build_desc_similarity_graph(self, top_k=10, threshold=0.0):
        desc_vectors = []
        embed_weight = self.embed.weight.data.cpu()
        embed_dim = embed_weight.size(1)

        for idx in range(self.Y):
            code = self.dicts["ind2c"][idx]
            desc = self.dicts["desc"].get(code, "")
            tokens = desc.lower().split()
            token_indices = [
                self.dicts["w2ind"].get(token, self.unk_idx)
                for token in tokens
                if token
            ]
            if not token_indices:
                desc_vectors.append(torch.zeros(embed_dim))
                continue
            inds = torch.tensor(token_indices, dtype=torch.long)
            desc_vectors.append(embed_weight[inds].mean(dim=0))

        desc_matrix = torch.stack(desc_vectors)
        desc_matrix = F.normalize(desc_matrix, p=2, dim=1)
        desc_matrix[torch.isnan(desc_matrix)] = 0.0

        similarity = torch.matmul(desc_matrix, desc_matrix.t())
        similarity[torch.isnan(similarity)] = 0.0
        similarity.fill_diagonal_(0.0)

        if threshold > 0:
            similarity = torch.where(
                similarity > threshold,
                similarity,
                torch.zeros_like(similarity),
            )

        if top_k and top_k < similarity.size(1):
            values, indices = torch.topk(similarity, k=top_k, dim=1)
            mask = torch.zeros_like(similarity)
            mask.scatter_(1, indices, values)
            similarity = mask

        similarity = torch.maximum(similarity, similarity.t())
        return similarity

    def _normalize_adjacency(self, adjacency):
        adjacency = adjacency + torch.eye(adjacency.size(0))
        degree = adjacency.sum(dim=1)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
        return (
            degree_inv_sqrt.unsqueeze(1)
            * adjacency
            * degree_inv_sqrt.unsqueeze(0)
        )

    def _maybe_edge_dropout(self, adjacency):
        if not self.training or self.label_edge_dropout <= 0:
            return adjacency
        keep_prob = 1.0 - self.label_edge_dropout
        mask = torch.rand_like(adjacency) < keep_prob
        dropped = adjacency * mask.float()
        return dropped

    def _propagate_label_graph(self, use_dropout=False):
        h = self.label_features

        if self.label_encoder_type == "gps":
            adj = self.label_adj
            if use_dropout:
                adj = self._maybe_edge_dropout(adj)
            for layer in self.graph_layers:
                proj = layer["proj"]
                block = layer["block"]
                if (
                    self.label_stochastic_depth > 0
                    and self.training
                    and torch.rand(1).item() < self.label_stochastic_depth
                ):
                    # Even if we skip the block, align dims if a proj exists
                    if proj is not None:
                        h = proj(h)
                    continue
                if proj is not None:
                    h = proj(h)
                h = block(h, adj)
            return h

        if self.label_encoder_type == "dir":
            adj_in = self.label_adj_in
            adj_out = self.label_adj_out
            if use_dropout:
                adj_in = self._maybe_edge_dropout(adj_in)
                adj_out = self._maybe_edge_dropout(adj_out)
            for layer in self.graph_layers:
                if (
                    self.label_stochastic_depth > 0
                    and self.training
                    and torch.rand(1).item() < self.label_stochastic_depth
                ):
                    continue
                h = layer(h, adj_in, adj_out)
            return h

        adj = self.label_adj
        if use_dropout:
            adj = self._maybe_edge_dropout(adj)
        for layer in self.graph_layers:
            if (
                self.label_stochastic_depth > 0
                and self.training
                and torch.rand(1).item() < self.label_stochastic_depth
            ):
                continue
            h = layer(h, adj)
        return h

    def _get_classifier_weights(self):
        if self._cached_classifier_weights is not None:
            return self._cached_classifier_weights
        states = self._propagate_label_graph()
        self._cached_label_states = states
        self._cached_classifier_weights = self.cls_proj(states)
        return self._cached_classifier_weights

    def forward(
        self,
        text,
        categorical=None,
        numerical=None,
        categorical_inputs=None,
        numerical_inputs=None,
        **_,
    ):
        self._cached_label_states = None
        self._cached_classifier_weights = None
        self.alpha = None

        # Backward-compatible arg handling: allow either categorical/numerical or *_inputs
        if categorical_inputs is None:
            categorical_inputs = categorical
        if numerical_inputs is None:
            numerical_inputs = numerical

        x = self.embed(text)
        x = self.embed_drop(x)
        x = x.transpose(1, 2)

        x = torch.tanh(self.conv(x).transpose(1, 2))

        label_states = self._propagate_label_graph(use_dropout=True)
        self._cached_label_states = label_states
        label_queries_base = self.attn_proj(label_states)
        label_classifiers = self.cls_proj(label_states)
        self._cached_classifier_weights = label_classifiers

        if self.use_text_summary_attn:
            text_mean = x.mean(dim=1)
            text_max, _ = x.max(dim=1)
            text_summary = torch.cat([text_mean, text_max], dim=1)
            text_query_bias = self.text_summary_proj(text_summary)
            label_queries = label_queries_base.unsqueeze(0) + text_query_bias.unsqueeze(1)
        else:
            label_queries = label_queries_base.unsqueeze(0)

        label_classifiers_exp = label_classifiers.unsqueeze(0)
        self.alpha = F.softmax(
            torch.matmul(label_queries, x.transpose(1, 2)), dim=2
        )
        m = self.alpha.matmul(x)
        y = label_classifiers_exp.mul(m).sum(dim=2).add(self.classifier_bias)

        return y

    def regularizer(self, labels=None):
        if not self.config.lmbda:
            return 0.0

        desc_vecs = []
        for label in labels:
            desc_vecs.append(
                [self.dv_dict[i] for i, l in enumerate(label) if l]
            )
        desc_data = [np.array(pad_desc_vecs(dvs)) for dvs in desc_vecs]

        b_batch = self.embed_descriptions(desc_data)
        classifier_weights = self._get_classifier_weights()

        diffs = []
        for i, bi in enumerate(b_batch):
            if not bi:
                continue
            ti = labels[i]
            inds = torch.nonzero(ti.data).squeeze().cpu().numpy()
            zi = classifier_weights[inds, :]
            diff = (zi - bi).mul(zi - bi).mean()
            diffs.append(self.config.lmbda * diff * bi.size()[0])

        if not diffs:
            return torch.tensor(0.0, device=labels.device)

        diff = torch.stack(diffs).mean()
        return diff

    def get_input_attention(self):
        if self.alpha is None:
            raise RuntimeError("Run a forward pass before requesting attention.")
        return self.alpha[:, :, :-1].cpu().detach().numpy()

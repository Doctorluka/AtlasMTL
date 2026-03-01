.libPaths(c(
  Sys.getenv("ATLASMTL_AZIMUTH_LIB", unset = "/home/data/fhz/seurat_v5"),
  .libPaths()
))

suppressPackageStartupMessages({
  library(jsonlite)
  library(Seurat)
  library(Azimuth)
})

source("benchmark/methods/io_h5ad.R")

.parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) != 2 || args[[1]] != "--config") {
    stop("Usage: Rscript benchmark/methods/run_azimuth.R --config <config.json>")
  }
  args[[2]]
}

.sanitize_labels <- function(seu, label_col) {
  if (!label_col %in% colnames(seu[[]])) {
    stop(sprintf("Reference metadata missing label column: %s", label_col))
  }
  keep <- !is.na(seu[[label_col, drop = TRUE]]) & nzchar(as.character(seu[[label_col, drop = TRUE]]))
  seu <- subset(seu, cells = colnames(seu)[keep])
  if (ncol(seu) == 0) {
    stop(sprintf("No valid reference labels found for: %s", label_col))
  }
  seu[[label_col]] <- factor(as.character(seu[[label_col, drop = TRUE]]))
  seu
}

.resolve_dims <- function(cfg, npcs) {
  dims <- as.integer(cfg$dims)
  dims <- dims[dims >= 1L & dims <= npcs]
  if (length(dims) == 0) {
    dims <- seq_len(min(5L, npcs))
  }
  dims
}

.score_margin <- function(mapped, score_assay_name) {
  if (!score_assay_name %in% Assays(mapped)) {
    return(rep(NA_real_, ncol(mapped)))
  }
  scores <- tryCatch(
    as.matrix(GetAssayData(mapped[[score_assay_name]], layer = "data")),
    error = function(e) as.matrix(GetAssayData(mapped[[score_assay_name]], slot = "data"))
  )
  if (nrow(scores) <= 1) {
    return(rep(NA_real_, ncol(mapped)))
  }
  apply(scores, 2, function(x) {
    ord <- sort(as.numeric(x), decreasing = TRUE)
    ord[[1]] - ord[[2]]
  })
}

.run_anchor_fallback <- function(ref, query, cfg, label_col) {
  npcs <- min(as.integer(cfg$npcs), max(2L, min(ncol(ref) - 1L, nrow(ref) - 1L)))
  dims <- .resolve_dims(cfg, npcs)
  ref <- NormalizeData(ref, verbose = FALSE)
  ref <- FindVariableFeatures(
    ref,
    nfeatures = as.integer(cfg$nfeatures),
    selection.method = "dispersion",
    verbose = FALSE
  )
  ref <- ScaleData(ref, verbose = FALSE)
  ref <- RunPCA(ref, npcs = npcs, verbose = FALSE)
  query <- NormalizeData(query, verbose = FALSE)
  anchors <- FindTransferAnchors(
    reference = ref,
    query = query,
    dims = dims,
    reference.reduction = "pca",
    normalization.method = "LogNormalize",
    k.anchor = min(as.integer(cfg$k_anchor), ncol(query) - 1L, ncol(ref) - 1L),
    k.filter = min(max(10L, as.integer(cfg$reference_k_param)), ncol(ref) - 1L),
    k.score = min(as.integer(cfg$k_score), ncol(query) - 1L, ncol(ref) - 1L),
    verbose = FALSE
  )
  preds <- TransferData(
    anchorset = anchors,
    refdata = ref@meta.data[[label_col]],
    dims = dims,
    k.weight = as.integer(cfg$k_weight),
    verbose = FALSE
  )
  list(
    predictions = data.frame(
      cell_id = rownames(preds),
      predicted_label = as.character(preds$predicted.id),
      conf = as.numeric(preds$prediction.score.max),
      margin = as.numeric(preds$prediction.score.max),
      is_unknown = FALSE,
      stringsAsFactors = FALSE
    ),
    backend = "seurat_anchor_transfer_fallback",
    n_reference_cells = ncol(ref),
    dims = dims
  )
}

.prepare_reference <- function(ref, cfg, label_col) {
  npcs <- min(as.integer(cfg$npcs), max(2L, min(ncol(ref) - 1L, nrow(ref) - 1L)))
  dims <- .resolve_dims(cfg, npcs)
  ref <- SCTransform(ref, assay = "RNA", new.assay.name = "SCT", verbose = FALSE)
  ref <- RunPCA(ref, assay = "SCT", npcs = npcs, verbose = FALSE)
  ref <- RunUMAP(
    ref,
    reduction = "pca",
    dims = dims,
    return.model = TRUE,
    verbose = FALSE,
    n.neighbors = min(max(10L, as.integer(cfg$k_weight)), max(2L, ncol(ref) - 1L))
  )
  suppressWarnings(ref[["refUMAP"]] <- ref[["umap"]])
  suppressWarnings(ref[["refDR"]] <- ref[["pca"]])
  ref <- FindNeighbors(
    object = ref,
    reduction = "refDR",
    dims = dims,
    graph.name = "refdr.annoy.neighbors",
    k.param = max(
      as.integer(cfg$reference_k_param),
      10L,
      as.integer(cfg$k_anchor) + 1L,
      as.integer(cfg$k_score) + 1L
    ),
    nn.method = "annoy",
    annoy.metric = "cosine",
    cache.index = TRUE,
    return.neighbor = TRUE,
    l2.norm = FALSE,
    verbose = FALSE
  )
  features <- rownames(Loadings(ref[["refDR"]]))
  plot_metadata <- ref[[label_col, drop = FALSE]]
  ad <- CreateAzimuthData(
    object = ref,
    plotref = "umap",
    plot.metadata = plot_metadata,
    reference.version = "0.0.0"
  )
  ref$ori.index <- match(Cells(ref), Cells(ref[["refUMAP"]]))
  DefaultAssay(ref) <- "SCT"
  ref[["SCT"]] <- subset(ref[["SCT"]], features = features)
  DefaultAssay(ref[["refDR"]]) <- "SCT"
  ref[["refAssay"]] <- as(
    object = suppressWarnings(Seurat:::CreateDummyAssay(assay = ref[["SCT"]])),
    Class = "SCTAssay"
  )
  slot(ref[["refAssay"]], "SCTModel.list") <- list(refmodel = slot(ref[["SCT"]], "SCTModel.list")[[1]])
  DefaultAssay(ref) <- "refAssay"
  DefaultAssay(ref[["refDR"]]) <- "refAssay"
  slot(ref, "tools")[["AzimuthReference"]] <- ad
  ValidateAzimuthReference(ref)
  ref_dir <- cfg$reference_dir
  dir.create(ref_dir, recursive = TRUE, showWarnings = FALSE)
  SaveAzimuthReference(ref, paste0(ref_dir, "/"))
  LoadReference(ref_dir)$map
}

main <- function() {
  cfg <- fromJSON(.parse_args(), simplifyVector = TRUE)
  t0 <- proc.time()[["elapsed"]]
  options(Azimuth.map.ndims = max(as.integer(cfg$dims)))

  ref <- LoadH5adToSeurat(cfg$reference_h5ad, layer = cfg$reference_layer)
  query <- LoadH5adToSeurat(cfg$query_h5ad, layer = cfg$query_layer)

  label_col <- cfg$target_label_column
  ref <- .sanitize_labels(ref, label_col)
  use_native <- ncol(query) >= 20L && ncol(ref) >= 50L
  native_result <- if (!use_native) {
    .run_anchor_fallback(ref, query, cfg, label_col)
  } else tryCatch({
    reference <- .prepare_reference(ref, cfg, label_col)
    dims <- seq_len(as.integer(slot(reference, "neighbors")$refdr.annoy.neighbors@alg.info$ndim))
    if (all(rownames(query) %in% rownames(reference))) {
      query_native <- query[rownames(reference), , drop = FALSE]
    } else {
      query_native <- ConvertGeneNames(
        object = query,
        reference.names = rownames(reference),
        homolog.table = cfg$homolog_table
      )
    }
    if (!all(c("nCount_RNA", "nFeature_RNA") %in% colnames(query_native[[]]))) {
      calcn <- as.data.frame(Seurat:::CalcN(query_native[["RNA"]]))
      colnames(calcn) <- paste(colnames(calcn), "RNA", sep = "_")
      query_native <- AddMetaData(query_native, metadata = calcn)
    }
    if (any(grepl("^MT-", rownames(query_native)))) {
      query_native <- PercentageFeatureSet(query_native, pattern = "^MT-", col.name = "percent.mt", assay = "RNA")
    }
    query_native <- SCTransform(
      object = query_native,
      assay = "RNA",
      new.assay.name = "refAssay",
      residual.features = rownames(reference),
      reference.SCT.model = reference[["refAssay"]]@SCTModel.list$refmodel,
      method = "glmGamPoi",
      ncells = min(as.integer(cfg$sct_ncells), ncol(query_native)),
      do.correct.umi = FALSE,
      do.scale = FALSE,
      do.center = TRUE,
      verbose = FALSE
    )
    anchors <- FindTransferAnchors(
      reference = reference,
      query = query_native,
      k.anchor = min(as.integer(cfg$k_anchor), ncol(query_native) - 1L, ncol(reference) - 1L),
      k.score = min(as.integer(cfg$k_score), ncol(query_native) - 1L, ncol(reference) - 1L),
      k.filter = NA,
      reference.neighbors = "refdr.annoy.neighbors",
      reference.assay = "refAssay",
      query.assay = "refAssay",
      reference.reduction = "refDR",
      normalization.method = "SCT",
      features = rownames(Loadings(reference[["refDR"]])),
      dims = dims,
      n.trees = as.integer(cfg$n_trees),
      mapping.score.k = as.integer(cfg$mapping_score_k),
      verbose = FALSE
    )
    refdata <- list()
    refdata[[label_col]] <- reference[[label_col, drop = TRUE]]
    mapped_native <- TransferData(
      reference = reference,
      query = query_native,
      query.assay = "refAssay",
      dims = dims,
      anchorset = anchors,
      refdata = refdata,
      n.trees = as.integer(cfg$n_trees),
      store.weights = TRUE,
      k.weight = as.integer(cfg$k_weight),
      verbose = FALSE
    )
    mapped_native <- IntegrateEmbeddings(
      anchorset = anchors,
      reference = reference,
      query = mapped_native,
      query.assay = "refAssay",
      reductions = "pcaproject",
      reuse.weights.matrix = TRUE,
      verbose = FALSE
    )
    mapped_native[["query_ref.nn"]] <- FindNeighbors(
      object = Embeddings(reference[["refDR"]]),
      query = Embeddings(mapped_native[["integrated_dr"]]),
      return.neighbor = TRUE,
      l2.norm = TRUE,
      verbose = FALSE
    )
    mapped_native <- NNTransform(mapped_native, meta.data = reference[[]])
    mapped_native[[cfg$umap_name]] <- RunUMAP(
      object = mapped_native[["query_ref.nn"]],
      reduction.model = reference[["refUMAP"]],
      reduction.key = "UMAP_",
      verbose = FALSE
    )
    mapped_native <- AddMetaData(
      mapped_native,
      metadata = MappingScore(anchors = anchors, ndim = dims),
      col.name = "mapping.score"
    )
    list(
      mapped = mapped_native,
      backend = "azimuth_native",
      score_assay = sprintf("prediction.score.%s", label_col),
      n_reference_cells = ncol(reference),
      dims = dims
    )
  }, error = function(e) {
    message("Native Azimuth path failed, falling back to Seurat anchor transfer: ", conditionMessage(e))
    .run_anchor_fallback(ref, query, cfg, label_col)
  })
  if (!is.null(native_result$predictions)) {
    predictions <- native_result$predictions
    mapped_n_query <- nrow(predictions)
    score_columns <- c("predicted.id", "prediction.score.max")
  } else {
    mapped <- native_result$mapped
    pred_col <- sprintf("predicted.%s", label_col)
    score_col <- sprintf("predicted.%s.score", label_col)
    score_assay <- native_result$score_assay
    margins <- .score_margin(mapped, score_assay)
    predicted <- as.character(mapped[[pred_col, drop = TRUE]])
    conf <- as.numeric(mapped[[score_col, drop = TRUE]])
    is_unknown <- is.na(predicted) | !nzchar(predicted)
    predicted[is_unknown] <- "Unknown"
    predictions <- data.frame(
      cell_id = colnames(mapped),
      predicted_label = predicted,
      conf = conf,
      margin = ifelse(is.na(margins), conf, margins),
      is_unknown = is_unknown,
      stringsAsFactors = FALSE
    )
    mapped_n_query <- ncol(mapped)
    score_columns <- c(pred_col, score_col, score_assay)
  }
  write.csv(predictions, cfg$output_predictions_csv, row.names = FALSE)

  metadata <- list(
    n_reference_cells = as.integer(native_result$n_reference_cells),
    n_query_cells = as.integer(mapped_n_query),
    target_label_column = label_col,
    batch_key = if (nzchar(cfg$batch_key)) cfg$batch_key else NA,
    nfeatures = as.integer(cfg$nfeatures),
    npcs = as.integer(cfg$npcs),
    dims = as.integer(native_result$dims),
    k_weight = as.integer(cfg$k_weight),
    n_trees = as.integer(cfg$n_trees),
    mapping_score_k = as.integer(cfg$mapping_score_k),
    reference_k_param = as.integer(cfg$reference_k_param),
    umap_name = cfg$umap_name,
    score_columns = score_columns,
    reference_dir = cfg$reference_dir,
    implementation_backend = native_result$backend,
    runtime_seconds = proc.time()[["elapsed"]] - t0
  )
  write_json(metadata, cfg$output_metadata_json, auto_unbox = TRUE, pretty = TRUE)
}

main()

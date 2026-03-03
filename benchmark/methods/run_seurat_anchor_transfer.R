.libPaths(c(
  Sys.getenv("ATLASMTL_SEURAT_LIB", unset = "/home/data/fhz/seurat_v5"),
  .libPaths()
))

suppressPackageStartupMessages({
  library(jsonlite)
  library(Seurat)
})

source("benchmark/methods/io_h5ad.R")

.parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) != 2 || args[[1]] != "--config") {
    stop("Usage: Rscript benchmark/methods/run_seurat_anchor_transfer.R --config <config.json>")
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

.run_umap_if_possible <- function(ref, dims) {
  tryCatch(
    list(
      reference = RunUMAP(ref, reduction = "pca", dims = dims, return.model = TRUE, verbose = FALSE),
      umap_available = TRUE
    ),
    error = function(e) {
      message("RunUMAP failed during Seurat reference preparation; falling back to TransferData-only path: ", conditionMessage(e))
      list(reference = ref, umap_available = FALSE)
    }
  )
}

.find_reference_features <- function(ref, cfg, batch_key, allow_umap = TRUE) {
  use_batches <- nzchar(batch_key) && batch_key %in% colnames(ref[[]])
  if (use_batches) {
    batch_values <- as.character(ref[[batch_key, drop = TRUE]])
    valid_batches <- batch_values[!is.na(batch_values) & nzchar(batch_values)]
    batch_counts <- table(valid_batches)
    use_batches <- length(batch_counts) > 1L &&
      min(as.integer(batch_counts)) >= 20L &&
      ncol(ref) >= 100L
  }
  base_npcs <- min(as.integer(cfg$npcs), max(2L, min(ncol(ref) - 1L, nrow(ref) - 1L)))
  dims <- .resolve_dims(cfg, base_npcs)
  if (!use_batches) {
    ref <- NormalizeData(ref, verbose = FALSE)
    ref <- FindVariableFeatures(
      ref,
      selection.method = "vst",
      nfeatures = as.integer(cfg$nfeatures),
      verbose = FALSE
    )
    features <- head(VariableFeatures(ref), as.integer(cfg$nfeatures))
    npcs <- min(base_npcs, max(1L, min(length(features) - 1L, ncol(ref) - 1L)))
    dims <- .resolve_dims(cfg, npcs)
    ref <- ScaleData(ref, features = features, verbose = FALSE)
    ref <- RunPCA(ref, features = features, npcs = npcs, verbose = FALSE)
    umap_result <- if (allow_umap) .run_umap_if_possible(ref, dims) else list(reference = ref, umap_available = FALSE)
    ref <- umap_result$reference
    return(list(
      reference = ref,
      dims = dims,
      features = features,
      reference_build_mode = "single_reference_pca",
      n_reference_batches = 1L,
      umap_available = umap_result$umap_available
    ))
  }

  tryCatch({
    obj.list <- SplitObject(ref, split.by = batch_key)
    obj.list <- lapply(obj.list, function(x) {
      x <- NormalizeData(x, verbose = FALSE)
      x <- FindVariableFeatures(
        x,
        selection.method = "vst",
        nfeatures = as.integer(cfg$nfeatures),
        verbose = FALSE
      )
      x
    })
    features <- SelectIntegrationFeatures(object.list = obj.list, nfeatures = as.integer(cfg$nfeatures))
    min_batch_cells <- min(vapply(obj.list, function(x) as.integer(ncol(x)), integer(1)))
    npcs <- min(base_npcs, max(1L, min(length(features) - 1L, min_batch_cells - 1L)))
    dims <- .resolve_dims(cfg, npcs)
    obj.list <- lapply(obj.list, function(x) {
      x <- ScaleData(x, features = features, verbose = FALSE)
      x <- RunPCA(x, features = features, npcs = npcs, verbose = FALSE)
      x
    })
    anchors <- FindIntegrationAnchors(
      object.list = obj.list,
      dims = dims,
      reduction = cfg$integration_reduction
    )
    ref <- IntegrateData(anchorset = anchors, dims = dims)
    DefaultAssay(ref) <- "integrated"
    ref <- ScaleData(ref, verbose = FALSE)
    ref <- RunPCA(ref, npcs = npcs, verbose = FALSE)
    umap_result <- if (allow_umap) .run_umap_if_possible(ref, dims) else list(reference = ref, umap_available = FALSE)
    ref <- umap_result$reference
    list(
      reference = ref,
      dims = dims,
      features = features,
      reference_build_mode = sprintf("split_%s_integration", cfg$integration_reduction),
      n_reference_batches = length(obj.list),
      umap_available = umap_result$umap_available
    )
  }, error = function(e) {
    message(
      "Reference integration path failed for Seurat anchor transfer; ",
      "falling back to single-reference PCA path: ",
      conditionMessage(e)
    )
    .find_reference_features(ref, cfg, batch_key = "", allow_umap = allow_umap)
  })
}

.transfer_without_umap <- function(ref, query, label_col, features, dims, normalization_method, cfg) {
  k.anchor <- min(as.integer(cfg$k_anchor), max(1L, ncol(query) - 1L), max(1L, ncol(ref) - 1L))
  k.score <- min(as.integer(cfg$k_score), max(1L, ncol(query) - 1L), max(1L, ncol(ref) - 1L))
  anchors <- FindTransferAnchors(
    reference = ref,
    query = query,
    dims = dims,
    features = features,
    reference.reduction = "pca",
    normalization.method = normalization_method,
    k.anchor = k.anchor,
    k.score = k.score,
    k.filter = min(max(5L, k.anchor + 1L), max(1L, ncol(ref) - 1L)),
    verbose = FALSE
  )
  preds <- TransferData(
    anchorset = anchors,
    refdata = ref@meta.data[[label_col]],
    dims = dims,
    k.weight = min(
      as.integer(cfg$k_weight),
      max(1L, ncol(ref) - 1L),
      max(1L, ncol(query) - 1L)
    ),
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
    backend = "seurat_anchor_transfer_transferdata",
    score_columns = c("predicted.id", "prediction.score.max")
  )
}

main <- function() {
  cfg <- fromJSON(.parse_args(), simplifyVector = TRUE)
  t0 <- proc.time()[["elapsed"]]

  ref <- LoadH5adToSeurat(cfg$reference_h5ad, layer = cfg$reference_layer)
  query <- LoadH5adToSeurat(cfg$query_h5ad, layer = cfg$query_layer)

  label_col <- cfg$target_label_column
  ref <- .sanitize_labels(ref, label_col)
  allow_umap <- ncol(ref) >= 50L && ncol(query) >= 20L
  prepared <- .find_reference_features(ref, cfg, cfg$batch_key, allow_umap = allow_umap)
  ref <- prepared$reference
  dims <- prepared$dims
  features <- prepared$features

  query <- NormalizeData(query, verbose = FALSE)
  transfer_result <- if (!isTRUE(prepared$umap_available)) {
    .transfer_without_umap(ref, query, label_col, features, dims, cfg$normalization_method, cfg)
  } else tryCatch({
    k.anchor <- min(as.integer(cfg$k_anchor), max(1L, ncol(query) - 1L), max(1L, ncol(ref) - 1L))
    k.score <- min(as.integer(cfg$k_score), max(1L, ncol(query) - 1L), max(1L, ncol(ref) - 1L))
    anchors <- FindTransferAnchors(
      reference = ref,
      query = query,
      dims = dims,
      features = features,
      reference.reduction = "pca",
      normalization.method = cfg$normalization_method,
      k.anchor = k.anchor,
      k.score = k.score,
      k.filter = min(max(5L, k.anchor + 1L), max(1L, ncol(ref) - 1L)),
      verbose = FALSE
    )
    mapped <- MapQuery(
      anchorset = anchors,
      reference = ref,
      query = query,
      refdata = setNames(list(ref[[label_col, drop = TRUE]]), label_col),
      reference.reduction = "pca",
      reduction.model = "umap",
      verbose = FALSE
    )

    pred_col <- sprintf("predicted.%s", label_col)
    score_col <- sprintf("predicted.%s.score", label_col)
    score_assay <- sprintf("prediction.score.%s", label_col)
    margins <- .score_margin(mapped, score_assay)
    predicted <- as.character(mapped[[pred_col, drop = TRUE]])
    conf <- as.numeric(mapped[[score_col, drop = TRUE]])
    is_unknown <- is.na(predicted) | !nzchar(predicted)
    predicted[is_unknown] <- "Unknown"
    list(
      predictions = data.frame(
        cell_id = colnames(mapped),
        predicted_label = predicted,
        conf = conf,
        margin = ifelse(is.na(margins), conf, margins),
        is_unknown = is_unknown,
        stringsAsFactors = FALSE
      ),
      mapped = mapped,
      backend = "seurat_anchor_transfer",
      score_columns = c(pred_col, score_col, score_assay)
    )
  }, error = function(e) {
    message("MapQuery failed during Seurat anchor transfer; falling back to TransferData-only path: ", conditionMessage(e))
    .transfer_without_umap(ref, query, label_col, features, dims, cfg$normalization_method, cfg)
  })
  predictions <- transfer_result$predictions
  write.csv(predictions, cfg$output_predictions_csv, row.names = FALSE)

  if (nzchar(cfg$output_reference_rds)) {
    saveRDS(ref, cfg$output_reference_rds)
  }
  if (nzchar(cfg$output_mapped_query_rds) && !is.null(transfer_result$mapped)) {
    saveRDS(transfer_result$mapped, cfg$output_mapped_query_rds)
  }

  metadata <- list(
    n_reference_cells = as.integer(ncol(ref)),
    n_query_cells = as.integer(nrow(predictions)),
    target_label_column = label_col,
    batch_key = if (nzchar(cfg$batch_key)) cfg$batch_key else NA,
    nfeatures = as.integer(cfg$nfeatures),
    npcs = as.integer(cfg$npcs),
    dims = as.integer(dims),
    k_anchor = as.integer(cfg$k_anchor),
    k_score = as.integer(cfg$k_score),
    k_weight = as.integer(cfg$k_weight),
    integration_reduction = cfg$integration_reduction,
    normalization_method = cfg$normalization_method,
    reference_build_mode = prepared$reference_build_mode,
    n_reference_batches = as.integer(prepared$n_reference_batches),
    score_columns = transfer_result$score_columns,
    implementation_backend = transfer_result$backend,
    runtime_seconds = proc.time()[["elapsed"]] - t0
  )
  write_json(metadata, cfg$output_metadata_json, auto_unbox = TRUE, pretty = TRUE)
}

main()

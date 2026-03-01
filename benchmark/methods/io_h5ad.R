suppressPackageStartupMessages({
  library(Matrix)
  library(Seurat)
  library(SingleCellExperiment)
  library(jsonlite)
})

.atlasmtl_python <- function() {
  python_bin <- Sys.getenv("ATLASMTL_PYTHON", unset = "/home/data/fhz/.local/share/mamba/envs/atlasmtl-env/bin/python")
  if (!file.exists(python_bin)) {
    stop(sprintf("ATLASMTL_PYTHON does not exist: %s", python_bin))
  }
  python_bin
}

.helper_script <- function() {
  helper <- file.path(dirname(sys.frame(1)$ofile %||% "benchmark/methods/io_h5ad.R"), "io_h5ad_helper.py")
  if (!file.exists(helper)) {
    helper <- "benchmark/methods/io_h5ad_helper.py"
  }
  normalizePath(helper, mustWork = TRUE)
}

`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}

.run_helper <- function(args) {
  output <- system2(.atlasmtl_python(), args = c(.helper_script(), args), stdout = TRUE, stderr = TRUE)
  status <- attr(output, "status")
  if (!is.null(status) && status != 0) {
    stop(paste(output, collapse = "\n"))
  }
  invisible(output)
}

ReadH5adMatrix <- function(h5ad.file, layer = "counts", fallback_x = TRUE, keep_obsm = NULL) {
  bundle_dir <- tempfile("h5ad_bundle_")
  dir.create(bundle_dir, recursive = TRUE, showWarnings = FALSE)
  on.exit(unlink(bundle_dir, recursive = TRUE, force = TRUE), add = TRUE)

  args <- c(
    "export-h5ad",
    "--input", normalizePath(h5ad.file, mustWork = TRUE),
    "--output-dir", bundle_dir
  )
  if (!is.null(layer)) {
    args <- c(args, "--layer", layer)
  }
  if (!is.null(keep_obsm) && length(keep_obsm) > 0) {
    args <- c(args, "--obsm", keep_obsm)
  }
  .run_helper(args)

  counts <- readMM(file.path(bundle_dir, "matrix.mtx"))
  obs <- read.csv(file.path(bundle_dir, "obs.csv"), row.names = 1, check.names = FALSE)
  var <- read.csv(file.path(bundle_dir, "var.csv"), row.names = 1, check.names = FALSE)
  metadata <- fromJSON(file.path(bundle_dir, "metadata.json"))

  rownames(counts) <- rownames(obs)
  colnames(counts) <- rownames(var)
  obsm <- list()
  if (!is.null(metadata$obsm_written)) {
    for (key in metadata$obsm_written) {
      obsm[[key]] <- as.matrix(read.csv(file.path(bundle_dir, sprintf("obsm__%s.csv", key)), row.names = 1, check.names = FALSE))
    }
  }
  list(
    matrix = counts,
    obs = obs,
    var = var,
    obsm = obsm,
    metadata = metadata
  )
}

ReadH5adObsVar <- function(h5ad.file) {
  payload <- ReadH5adMatrix(h5ad.file, layer = NULL, fallback_x = TRUE, keep_obsm = NULL)
  list(obs = payload$obs, var = payload$var, metadata = payload$metadata)
}

LoadH5adToSeurat <- function(h5ad.file, keep_obsm = NULL, layer = "counts") {
  payload <- ReadH5adMatrix(h5ad.file, layer = layer, keep_obsm = keep_obsm)
  counts <- t(payload$matrix)
  seu <- Seurat::CreateSeuratObject(counts = counts, meta.data = payload$obs)
  if (!is.null(keep_obsm)) {
    for (obsm_name in names(payload$obsm)) {
      coords <- payload$obsm[[obsm_name]]
      coords <- coords[rownames(payload$obs), , drop = FALSE]
      colnames(coords) <- paste0(obsm_name, "_", seq_len(ncol(coords)))
      seu[[obsm_name]] <- Seurat::CreateDimReducObject(coords = coords, key = paste0(obsm_name, "_"))
    }
  }
  seu
}

LoadH5ad <- function(h5ad.file, keep_obsm = NULL, layers = "counts") {
  LoadH5adToSeurat(h5ad.file = h5ad.file, keep_obsm = keep_obsm, layer = layers)
}

LoadH5adToSCE <- function(h5ad.file, label_columns = NULL, layer = "counts", keep_obsm = NULL) {
  payload <- ReadH5adMatrix(h5ad.file, layer = layer, keep_obsm = keep_obsm)
  counts <- t(payload$matrix)
  sce <- SingleCellExperiment(
    assays = list(counts = as(counts, "dgCMatrix")),
    colData = S4Vectors::DataFrame(payload$obs),
    rowData = S4Vectors::DataFrame(payload$var)
  )
  if (!is.null(keep_obsm)) {
    for (obsm_name in names(payload$obsm)) {
      reducedDims(sce)[[obsm_name]] <- payload$obsm[[obsm_name]][colnames(sce), , drop = FALSE]
    }
  }
  sce
}

NormalizeLog1pMatrix <- function(mat, target_sum = 1e4) {
  lib_sizes <- Matrix::colSums(mat)
  lib_sizes[lib_sizes <= 0] <- 1
  scaled <- t(t(mat) / lib_sizes * target_sum)
  log1p(scaled)
}

WriteSeuratToH5ad <- function(seurat_obj, filename, obsm = c("umap", "pca"), layer_name = "counts") {
  assay_name <- DefaultAssay(seurat_obj)
  counts <- tryCatch(
    Seurat::GetAssayData(seurat_obj, assay = assay_name, slot = "counts"),
    error = function(e) NULL
  )
  if (is.null(counts) || length(counts) == 0) {
    counts <- Seurat::GetAssayData(seurat_obj, assay = assay_name, slot = "data")
  }
  if (is.null(counts) || length(counts) == 0) {
    stop("Unable to extract counts/data matrix from Seurat object")
  }
  obs <- seurat_obj@meta.data
  var <- data.frame(gene = rownames(counts), row.names = rownames(counts), check.names = FALSE)

  bundle_dir <- tempfile("seurat_h5ad_bundle_")
  dir.create(bundle_dir, recursive = TRUE, showWarnings = FALSE)
  on.exit(unlink(bundle_dir, recursive = TRUE, force = TRUE), add = TRUE)

  writeMM(as(counts, "dgCMatrix"), file.path(bundle_dir, "matrix.mtx"))
  write.csv(obs, file.path(bundle_dir, "obs.csv"))
  write.csv(var, file.path(bundle_dir, "var.csv"))

  if (!is.null(obsm)) {
    for (dimred in obsm) {
      if (!dimred %in% names(seurat_obj@reductions)) {
        next
      }
      coords <- seurat_obj@reductions[[dimred]]@cell.embeddings
      coords <- coords[rownames(obs), , drop = FALSE]
      write.csv(coords, file.path(bundle_dir, sprintf("obsm__%s.csv", dimred)))
    }
  }

  .run_helper(c(
    "import-h5ad",
    "--input-dir", bundle_dir,
    "--output", normalizePath(filename, mustWork = FALSE),
    "--layer-name", layer_name
  ))
  invisible(filename)
}

SeuratToAnndata <- function(seurat_obj, filename = NULL, obsm = c("umap", "pca")) {
  if (is.null(filename)) {
    stop("filename must be provided")
  }
  WriteSeuratToH5ad(seurat_obj = seurat_obj, filename = filename, obsm = obsm)
  invisible(filename)
}

-- What the stored index was built with (source file hash + embedding model + chunking settings +
-- parser/chunker versions). A change in any of them means the stored vectors and chunks are
-- stale even if the source file is the same, so the loader rebuilds instead of skipping.
-- Rows ingested before this migration have no fingerprint and are rebuilt once.
ALTER TABLE documents ADD COLUMN index_fingerprint text;

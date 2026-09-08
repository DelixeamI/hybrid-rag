CREATE EXTENSION IF NOT EXISTS vector;

-- One row per ingested edition of a corpus. A new edition (different sha256)
-- replaces the previous one with the same name; the same edition is a no-op.
CREATE TABLE documents (
    id          serial PRIMARY KEY,
    name        text        NOT NULL,
    sha256      text        NOT NULL,
    ingested_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (name, sha256)
);

CREATE TABLE chunks (
    id            bigserial PRIMARY KEY,
    document_id   integer NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    chunk_key     text    NOT NULL,
    article_number text   NOT NULL,
    article_title text    NOT NULL,
    section       text,
    chapter       text,
    part_numbers  text[]  NOT NULL DEFAULT '{}',
    breadcrumbs   text    NOT NULL,
    text          text    NOT NULL,
    token_count   integer NOT NULL,
    embedding     vector(384) NOT NULL,
    -- Title weighs more than body. 'russian' = stemming + stop words.
    tsv           tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('russian', article_title), 'A') ||
        setweight(to_tsvector('russian', text), 'B')
    ) STORED,
    UNIQUE (document_id, chunk_key)
);

CREATE INDEX chunks_tsv_gin ON chunks USING gin (tsv);
CREATE INDEX chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX chunks_document_idx ON chunks (document_id);

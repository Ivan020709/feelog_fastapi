-- DBeaver에서 feelog_rag 데이터베이스를 선택한 뒤 실행합니다.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS contract_chunk (
    id BIGSERIAL PRIMARY KEY,
    source VARCHAR(255) NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(1536) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_contract_chunk_embedding
ON contract_chunk USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

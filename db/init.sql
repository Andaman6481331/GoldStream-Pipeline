-- BRONZE
CREATE TABLE IF NOT EXISTS landing_ticks (
    id          BIGSERIAL PRIMARY KEY,
    symbol      VARCHAR(20),
    bid         DOUBLE PRECISION,
    ask         DOUBLE PRECISION,
    last        DOUBLE PRECISION,
    volume      DOUBLE PRECISION,
    time_msc    BIGINT,
    ingested_at TIMESTAMPTZ DEFAULT NOW()
);

-- SILVER
CREATE TABLE IF NOT EXISTS cleaned_ticks (
    id          BIGSERIAL PRIMARY KEY,
    symbol      VARCHAR(20),
    bid         DOUBLE PRECISION,
    ask         DOUBLE PRECISION,
    spread      DOUBLE PRECISION,
    rsi         DOUBLE PRECISION,
    ema_20      DOUBLE PRECISION,
    tick_time   TIMESTAMPTZ,
    cleaned_at  TIMESTAMPTZ DEFAULT NOW(),
    time_msc    BIGINT,
    CONSTRAINT unique_time_msc UNIQUE (time_msc)
);

-- GOLD
CREATE TABLE IF NOT EXISTS trade_decisions (
    id            BIGSERIAL PRIMARY KEY,
    symbol        VARCHAR(20),
    decision      VARCHAR(10),
    bid           DOUBLE PRECISION,
    ask           DOUBLE PRECISION,
    rsi           DOUBLE PRECISION,
    ema_20        DOUBLE PRECISION,
    spread        DOUBLE PRECISION,
    tick_time     TIMESTAMPTZ,
    decided_at    TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT unique_trade_tick_time UNIQUE (tick_time)
);
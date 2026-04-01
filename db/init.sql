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
    reason        VARCHAR(255),
    score         INTEGER,
    bid           DOUBLE PRECISION,
    ask           DOUBLE PRECISION,
    rsi           DOUBLE PRECISION,
    ema_20        DOUBLE PRECISION,
    spread        DOUBLE PRECISION,
    tick_time     TIMESTAMPTZ,
    decided_at    TIMESTAMPTZ DEFAULT NOW(),
    -- SMC Phase 2 context
    smc_trend_15m VARCHAR(10),
    bos_detected_15m BOOLEAN,
    choch_detected_15m BOOLEAN,
    market_bias_4h VARCHAR(10),
    liq_swept BOOLEAN,
    liq_side VARCHAR(10),
    CONSTRAINT unique_trade_tick_time UNIQUE (tick_time)
);
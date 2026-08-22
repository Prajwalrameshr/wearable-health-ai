from datetime import datetime, timezone
from sqlalchemy import BigInteger, Column, DateTime, Float, Integer, String, Text, UniqueConstraint
from database import Base


class HealthLog(Base):
    __tablename__ = "health_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    device_user_id = Column(String(100), index=True, nullable=False)
    record_date = Column(String(10), index=True, nullable=False)

    # Health Connect Raw Metrics
    steps = Column(BigInteger, default=0)
    heart_rate = Column(Float, nullable=True)
    oxygen_saturation = Column(Float, nullable=True)
    sleep_minutes = Column(BigInteger, default=0)

    record_start_time = Column(String(50), nullable=True)
    record_end_time = Column(String(50), nullable=True)
    collected_at = Column(String(50), nullable=True)

    # ML Prediction Outputs
    predicted_state = Column(String(50), nullable=True)  # Recovery, Baseline, Strain
    risk_score = Column(Float, nullable=True)
    risk_level = Column(String(20), nullable=True)        # Low, Moderate, High
    clinical_advisory_level = Column(String(30), nullable=True)
    clinical_summary_message = Column(Text, nullable=True)
    window_used = Column(String(20), default="none")       # 7_days, 30_days, none

    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        UniqueConstraint("device_user_id", "record_date", name="uq_user_daily_record"),
    )

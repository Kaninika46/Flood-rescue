"""Database tables (PostGIS, SRID 4326 longitude / latitude)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from geoalchemy2 import Geometry
from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Declarative base."""


class Scene(Base):
    """One Sentinel-1 scene / event run."""

    __tablename__ = "scenes"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    s1_path: Mapped[str] = mapped_column(Text)
    state: Mapped[str] = mapped_column(String(20), default="new")  # new/processing/done/failed/superseded
    minutes: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class FloodLayer(Base):
    """A raster produced for a scene (paths to GeoTIFFs, not the pixels)."""

    __tablename__ = "flood_layers"
    id: Mapped[int] = mapped_column(primary_key=True)
    scene_id: Mapped[int] = mapped_column(ForeignKey("scenes.id", ondelete="CASCADE"))
    kind: Mapped[str] = mapped_column(String(30))  # probability / confidence / depth / mask
    path: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class RoadStatus(Base):
    """Status of one directed road edge for a scene."""

    __tablename__ = "road_status"
    id: Mapped[int] = mapped_column(primary_key=True)
    scene_id: Mapped[int | None] = mapped_column(ForeignKey("scenes.id", ondelete="CASCADE"),
                                                 nullable=True)
    edge_id: Mapped[str] = mapped_column(String(80), index=True)
    status: Mapped[str] = mapped_column(String(20))
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_depth_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String(20), default="level1")
    bridge: Mapped[bool] = mapped_column(Boolean, default=False)
    geom: Mapped[Any] = mapped_column(Geometry("LINESTRING", srid=4326))


class Route(Base):
    """A computed route (stored for the dashboard and the app)."""

    __tablename__ = "routes"
    id: Mapped[int] = mapped_column(primary_key=True)
    vehicle: Mapped[str] = mapped_column(String(20))
    total_time_min: Mapped[float | None] = mapped_column(Float, nullable=True)
    n_verify: Mapped[int] = mapped_column(Integer, default=0)
    geojson: Mapped[dict[str, Any]] = mapped_column(JSON)
    geom: Mapped[Any] = mapped_column(Geometry("LINESTRING", srid=4326), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Priority(Base):
    """One ranked area."""

    __tablename__ = "priorities"
    id: Mapped[int] = mapped_column(primary_key=True)
    scene_id: Mapped[int | None] = mapped_column(ForeignKey("scenes.id", ondelete="CASCADE"),
                                                 nullable=True)
    priority: Mapped[int] = mapped_column(Integer)
    area_name: Mapped[str] = mapped_column(String(200))
    exposed_people: Mapped[float] = mapped_column(Float, default=0.0)
    flooded_km2: Mapped[float] = mapped_column(Float, default=0.0)
    cut_off: Mapped[bool] = mapped_column(Boolean, default=False)
    verify: Mapped[bool] = mapped_column(Boolean, default=False)
    mean_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    geom: Mapped[Any] = mapped_column(Geometry("MULTIPOLYGON", srid=4326))


class Alert(Base):
    """CAP alert with its approval trail; sendable only with approved_by and approved_at."""

    __tablename__ = "alerts"
    id: Mapped[int] = mapped_column(primary_key=True)
    scene_id: Mapped[int | None] = mapped_column(ForeignKey("scenes.id", ondelete="SET NULL"),
                                                 nullable=True)
    priority_id: Mapped[int | None] = mapped_column(ForeignKey("priorities.id"), nullable=True)
    identifier: Mapped[str] = mapped_column(String(120), unique=True)
    cap_xml: Mapped[str] = mapped_column(Text)
    sms: Mapped[str] = mapped_column(Text)
    state: Mapped[str] = mapped_column(String(20), default="draft")
    approved_by: Mapped[str | None] = mapped_column(String(120), nullable=True)
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    approved_role: Mapped[str | None] = mapped_column(String(20), nullable=True)  # token role
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    sent_count: Mapped[int] = mapped_column(Integer, default=0)  # numbers are never stored
    cancel_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    cancelled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class FieldReport(Base):
    """A report from a rescuer about one road edge."""

    __tablename__ = "field_reports"
    id: Mapped[int] = mapped_column(primary_key=True)
    edge_id: Mapped[str] = mapped_column(String(80), index=True)
    report: Mapped[str] = mapped_column(String(30))  # road_open / road_closed / bridge_gone
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    reporter: Mapped[str | None] = mapped_column(String(120), nullable=True)
    geom: Mapped[Any] = mapped_column(Geometry("POINT", srid=4326), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

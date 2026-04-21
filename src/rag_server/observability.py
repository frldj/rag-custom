import os
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter

def setup_observability():
    # Configuration des endpoints
    trace_endpoint = os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://localhost:4318/v1/traces")
    metric_endpoint = os.getenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", "http://localhost:4318/v1/metrics")

    # Trace Provider
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=trace_endpoint)))
    trace.set_tracer_provider(tracer_provider)

    # Metric Provider
    metric_reader = PeriodicExportingMetricReader(OTLPMetricExporter(endpoint=metric_endpoint))
    meter_provider = MeterProvider(metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)

    # Création du meter et des métriques
    meter = metrics.get_meter("rag.metrics")
    
    rag_metrics = {
        "retrieval_time": meter.create_histogram("retrieval_time_ms", description="Time to retrieve", unit="ms"),
        "reranker_time": meter.create_histogram("context_reranker_time_ms", description="Time to rerank", unit="ms"),
        "llm_ttft": meter.create_histogram("llm_ttft_ms", description="LLM TTFT", unit="ms"),
        "llm_generation_time": meter.create_histogram("llm_generation_time_ms", description="LLM Generation Time", unit="ms"),
        "rag_ttft": meter.create_histogram("rag_ttft_ms", description="RAG TTFT", unit="ms"),
        "input_tokens": meter.create_counter("input_tokens_total", description="Total Input tokens"),
        "output_tokens": meter.create_counter("output_tokens_total", description="Total Output tokens"),
        "total_tokens": meter.create_counter("total_tokens_total", description="Total tokens used"),
        "api_requests": meter.create_counter("api_requests_total", description="Total API requests"),
        "hallucinations": meter.create_counter("rag_hallucinations_total", description="Nombre d'hallucinations détectées"),
        "cache_hits": meter.create_counter("rag_cache_hits_total", description="Nombre de succès du cache Redis"),
        "cache_misses": meter.create_counter("rag_cache_miss_total", description="Nombre d'échecs du cache Redis"),
        "circuit_breaker_open": meter.create_counter("rag_circuit_breaker_open_total", description="Nombre d'ouvertures de circuit breaker")
    }
    
    return rag_metrics
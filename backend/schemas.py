

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
from uuid import uuid4


class ArticleMetadata(BaseModel):
    #Metadata extracted separately from article text.
    reading_time: Optional[int] = Field(None, ge=0)
    tags: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)

    @field_validator('tags', 'categories')
    def clean_list_items(v: List[str]) -> List[str]:
        return [item.strip() for item in v if item and item.strip()]


class ArticleContent(BaseModel):
    #Cleaned article content after scraping.
    url: str
    title: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    author: Optional[str] = None
    publish_date: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('url')
    
    def validate_url(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("URL cannot be empty")
        return v.strip()

    @field_validator('title', 'text')
    
    def validate_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()


class TextChunk(BaseModel):
    #Single semantic chunk of article text sent to the LLM.
    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    text: str = Field(..., min_length=1)
    position: int = Field(..., ge=0)
    token_count: int = Field(..., ge=0)
    source_url: str

    @field_validator('text')
    
    def validate_text(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Text chunk cannot be empty")
        return v.strip()


class Entity(BaseModel):
  #Graph node extracted from text.
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(..., min_length=1)
    type: str = Field(..., min_length=1)
    properties: Dict[str, Any] = Field(default_factory=dict)
    mentions: List[str] = Field(default_factory=list)

    @field_validator('name', 'type')
    
    def validate_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()


class Relationship(BaseModel):
    #Edge between two entities in the graph.
    id: str = Field(default_factory=lambda: str(uuid4()))
    source_entity_id: str = Field(..., min_length=1)
    target_entity_id: str = Field(..., min_length=1)
    relationship_type: str = Field(..., min_length=1)
    properties: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('source_entity_id', 'target_entity_id', 'relationship_type')
    
    def validate_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()

    @field_validator('target_entity_id')
    
    def validate_different_entities(cls, v: str, info) -> str:
        if 'source_entity_id' in info.data and v == info.data['source_entity_id']:
            raise ValueError("Source and target entities must be different")
        return v


class GraphExtraction(BaseModel):
    #Entities and relationships extracted from a single text chunk.
    entities: List[Entity] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)
    chunk_id: str


class GraphData(BaseModel):
    #Merged graph used for storage and visualization.
    nodes: List[Entity] = Field(default_factory=list)
    edges: List[Relationship] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('nodes')
    
    def validate_unique_nodes(cls, v: List[Entity]) -> List[Entity]:
        ids = [n.id for n in v]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate node IDs found")
        return v

    @field_validator('edges')
    def validate_edge_references(cls, v: List[Relationship], info) -> List[Relationship]:
        if 'nodes' in info.data:
            node_ids = {n.id for n in info.data['nodes']}
            for edge in v:
                if edge.source_entity_id not in node_ids:
                    raise ValueError(f"Invalid source node: {edge.source_entity_id}")
                if edge.target_entity_id not in node_ids:
                    raise ValueError(f"Invalid target node: {edge.target_entity_id}")
        return v


class GraphSchema(BaseModel):
    #Controls what the LLM is allowed to extract.
    entity_types: List[str] = Field(default_factory=list)
    relationship_types: List[str] = Field(default_factory=list)
    extraction_prompt: str = Field(..., min_length=1)

    @field_validator('entity_types', 'relationship_types')
    
    def validate_unique_types(cls, v: List[str]) -> List[str]:
        cleaned = [item.strip().upper() for item in v if item and item.strip()]
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("Duplicate types found")
        return cleaned

    @field_validator('extraction_prompt')
    def validate_prompt(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Extraction prompt cannot be empty")
        return v.strip()


class ScrapingFailure(BaseModel):
    #Logs scraping failures instead of crashing the pipeline.
    timestamp: datetime = Field(default_factory=datetime.now)
    url: str
    domain: str
    failure_type: str
    http_status: Optional[int] = None
    error_message: Optional[str] = None


__all__ = [
    'ArticleMetadata',
    'ArticleContent',
    'TextChunk',
    'Entity',
    'Relationship',
    'GraphExtraction',
    'GraphData',
    'GraphSchema',
    'ScrapingFailure'
]

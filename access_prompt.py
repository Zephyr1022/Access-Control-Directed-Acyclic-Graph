"""
Access Control Directed Acyclic Graph (DAG) to a Knowledge Graph

This module provides prompt engineering and message generation for three main approaches
to extract knowledge graphs from Access Control DAG (Directed Acyclic Graph) images:

1. Entity Extraction: Extract entities (nodes) from image
2. Relation Classification: Binary classification of relations between two entities
3. Path Enumeration: Extract all paths and relations in the graph
"""

import json
import base64
from collections import deque
from typing import Dict, List, Optional, Any, Tuple

# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

# Few-shot example paths
FEW_SHOT_DIR = "/home/xingmeng/access_control/data/SubgraphsWithTriples/one-shot"
FEW_SHOT_BASE = "enterprise_clients_graph_policies_graph_part1__association_client_organization_b_to_application_services"
FEW_SHOT_JSON_PATH = f"{FEW_SHOT_DIR}/{FEW_SHOT_BASE}.json"
FEW_SHOT_IMAGE_PATH = f"{FEW_SHOT_DIR}/{FEW_SHOT_BASE}_labeled.png"
FEW_SHOT_IMAGE_B_PATH = f"{FEW_SHOT_DIR}/{FEW_SHOT_BASE}_labeled_b.png"

# Node types Reference
NODE_TYPES = {
    "user_attributes": "blue/cyan nodes representing people, groups, or organizational units",
    "object_attributes": "green nodes representing resources, data, systems, or infrastructure",
    "policy_classes": "red/orange nodes representing top-level containers (incoming edges only, no outgoing edges)"
}

# Relationship types and their visual characteristics
RELATIONSHIP_TYPES = {
    "assign": "solid black arrows (→) for hierarchical assignment relationships",
    "permit": "green arrows (→) for permission/association relationships with weights",
    "prohibit": "red dashed arrows (⇢) for prohibition/denial relationships"
}

# System prompt for vision tasks
SYSTEM_PROMPT_VISION = """You are an expert at analyzing images and understanding visual diagrams.
You can see and interpret images, graphs, charts, and visual representations.
You provide accurate, detailed analysis based on what you observe in the images."""

# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

ENTITY_EXTRACTION_PROMPT = """Identify and extract ALL entities from the provided Access Control DAG (Directed Acyclic Graph) image.

Task:
1. Locate every node in the graph by identifying rectangular boxes with text labels.
2. Assign sequential IDs (n1, n2, n3, etc.) to each node.
3. Categorize each node into exactly one of these three types based on color and position:
   - "user_attributes" (blue/cyan nodes): People, groups, or organizational units (typically sources of arrows, leftmost/top position)
   - "object_attributes" (green nodes): Resources, data, systems, or infrastructure (typically destinations of arrows, middle position)
   - "policy_classes" (red/orange nodes): Top-level containers where paths converge (incoming edges only, NO outgoing edges, rightmost/bottom position)
4. Extract the exact text label for each node.

Instructions:
- Be comprehensive: Include every visible node with a text label.
- Use visual cues: Node background colors are key indicators:
  * Blue/cyan boxes = user_attributes (e.g., "developers", "system_administrators")
  * Green boxes = object_attributes (e.g., "application_services", "apis")
  * Red/orange boxes = policy_classes (e.g., "enterprise_clients_graph_policies")
- Use graph structure: policy_classes have incoming arrows but NO outgoing arrows.
- Be precise: Use the exact naming shown in the image text labels.
- Flow pattern: Look for left-to-right or top-to-bottom flow patterns.

Examples:
- If you see a blue box labeled "client_organization_b" → classify as user_attributes
- If you see a green box labeled "application_services" → classify as object_attributes
- If you see a red box labeled "enterprise_clients_graph_policies" with only incoming arrows → classify as policy_classes

Output Format (Strict JSON only):
{
  "nodes": [
    {"node_id": "n1", "type": "user_attributes", "content": "label_name"},
    {"node_id": "n2", "type": "object_attributes", "content": "label_name"},
    {"node_id": "n3", "type": "policy_classes", "content": "label_name"}
    ...
  ]
}"""


RELATION_CLASSIFICATION_PROMPT = """Determine if a direct directed relationship exists between two specific entities in the provided Access Control DAG image.

Task: Check for a direct "{relation_type}" arrow from "{from_entity}" to "{to_entity}".

Entities to Validate:
- Source: {from_entity}
- Target: {to_entity}
- Relationship Direction: {from_entity} → {to_entity}

Visual Relationship Types:
- "assign": Solid black arrows (→) connecting nodes in hierarchical flow
- "permit": Green arrows (→) indicating permission/association with possible weight labels
- "prohibit": Red dashed arrows (⇢) indicating denial/prohibition

Decision Criteria:
- Answer "Yes" ONLY if you see the correct arrow TYPE and STYLE:
  - For "assign": Look for solid black arrows (→)
  - For "permit": Look for green arrows (→) (may have text labels like "read", "deploy")
  - For "prohibit": Look for red dashed arrows (⇢)
- Answer "No" if:
  - The arrow direction is reversed ({to_entity} → {from_entity}).
  - The connection is indirect (through intermediate nodes).
  - Wrong arrow color/style (e.g., green arrow for "assign" query).
  - No arrow exists between these two entities.
  - One or both entities are not visible in the image.

Examples:
- If querying "assign" between "developers" and "client_organization_b": Look for solid black arrow developers → client_organization_b
- If querying "permit" between "client_organization_b" and "application_services": Look for green arrow with possible "read, deploy" labels
- If querying "prohibit" between nodes: Look for red dashed arrow (⇢)

Output Format (Strict JSON only):
{{
  "entity1": "{from_entity}",
  "entity2": "{to_entity}",
  "relation": "{relation_type}",
  "exists": "Yes" or "No",
  "confidence": "high", "medium", or "low",
  "explanation": "Brief visual evidence from the image (specify arrow color/style if found)",
  "subrelations": ["list", "of", "weights", "if", "any"]
}}"""


PATH_ENUMERATION_PROMPT = """Extract all relation triples and paths from the Access Control DAG image.

Task:
1. Identify all nodes by color and assign sequential IDs (n1, n2, etc.):
   - user_attributes: blue/cyan nodes (sources)
   - object_attributes: green nodes (resources)
   - policy_classes: red/orange nodes (sinks, no outgoing arrows)
2. Identify all direct edges (arrows) between nodes using visual characteristics:
   - "assign": Solid black arrows (→) - hierarchical assignments
   - "permit": Green arrows (→) - permissions with possible weight labels ("read", "write", "deploy")
   - "prohibit": Red dashed arrows (⇢) - prohibitions/denials
3. For "permit" relations, extract subrelations (permission weights) from text labels on arrows.
4. Enumerate all sequential paths from user_attributes nodes to policy_classes nodes.

Instructions:
- Visual identification: Use node colors and arrow colors/styles as primary identification method:
  * Blue nodes = user_attributes, Green nodes = object_attributes, Red nodes = policy_classes
  * Black solid arrows = "assign", Green arrows = "permit", Red dashed arrows = "prohibit"
- Observe only: Only include nodes and arrows PHYSICALLY VISIBLE in the image.
- Sequential logic: Trace paths node-by-node following arrow directions and colors.
- Arrow specificity: Distinguish between solid black (assign), green (permit), and red dashed (prohibit) arrows.
- Exactness: Use exact labels from image text, preserve arrow colors in descriptions.

Examples:
- Path: blue "developers" → black arrow → blue "client_organization_b" → green arrow → green "application_services" → black arrow → red "enterprise_clients_graph_policies"
- Relationship identification: Green arrow with "read, deploy" text = "permit" with subrelations ["read", "deploy"]
- Node sequence: Always follows user_attributes → object_attributes → policy_classes pattern

Output Format (Strict JSON only):
{{
  "nodes": [
    {{"node_id": "n1", "type": "user_attributes", "content": "label"}},
    ...
  ],
  "edges": [
    {{
      "from_id": "n1",
      "source_name": "src",
      "to_id": "n2",
      "target_name": "tgt",
      "relationship": "permit",
      "subrelations": ["read", "deploy"]
    }}
  ],
  "paths": [
    {{
      "path_id": "path_1",
      "nodes": ["n1", "n2", "n3"],
      "relationships": ["assign", "permit"],
      "length": 2,
      "description": "Sequential path from n1 to n3"
    }}
  ],
  "graph_metadata": {{
    "total_nodes": 0,
    "total_edges": 0,
    "total_paths": 0
  }}
}}"""

# ============================================================================
# MESSAGE GENERATION FUNCTIONS
# ============================================================================


def get_entity_extraction_messages(image_base64: str, filename: str, few_shot_examples: Optional[List[Dict]] = None) -> List[Dict]:
    """
    Generate messages for entity extraction from Access Control DAG images.

    Args:
        image_base64: Base64 encoded image data
        filename: Image filename (unused in current implementation)
        few_shot_examples: Optional few-shot example messages for Context7 sequential processing

    Returns:
        Complete message sequence for entity extraction task
    """
    return build_message_sequence(ENTITY_EXTRACTION_PROMPT, image_base64, few_shot_examples)


def get_relation_classification_messages(
    image_base64: str,
    triple_data: Dict[str, str],
    few_shot_examples: Optional[List[Dict]] = None
) -> List[Dict]:
    """
    Generate messages for binary relation classification between two entities.

    Args:
        image_base64: Base64 encoded image data
        triple_data: Dictionary with 'from_entity', 'to_entity', and 'relationship' keys
        few_shot_examples: Optional few-shot example messages for Context7 sequential processing

    Returns:
        Complete message sequence for relation classification task
    """
    from_entity = triple_data['from_entity']
    to_entity = triple_data['to_entity']
    relation_type = triple_data.get('relationship', 'assign')

    prompt = RELATION_CLASSIFICATION_PROMPT.format(
        from_entity=from_entity,
        to_entity=to_entity,
        relation_type=relation_type
    )

    return build_message_sequence(prompt, image_base64, few_shot_examples)


def get_path_enumeration_messages(image_base64: str, filename: str, few_shot_examples: Optional[List[Dict]] = None) -> List[Dict]:
    """
    Generate messages for path enumeration and graph extraction from Access Control DAG images.

    Args:
        image_base64: Base64 encoded image data
        filename: Image filename (unused in current implementation)
        few_shot_examples: Optional few-shot example messages for Context7 sequential processing

    Returns:
        Complete message sequence for path enumeration task
    """
    return build_message_sequence(PATH_ENUMERATION_PROMPT, image_base64, few_shot_examples)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def convert_ground_truth_to_entity_extraction_format(ground_truth_json: Dict) -> Dict:
    """
    Convert ground truth JSON to entity extraction output format.
    
    Args:
        ground_truth_json: Ground truth JSON with policy_elements structure
        
    Returns:
        Dictionary with nodes array in entity extraction format
    """
    nodes = []
    node_id_counter = 1
    
    # Extract user_attributes
    if "policy_elements" in ground_truth_json:
        policy_elements = ground_truth_json["policy_elements"]
        
        # Add user_attributes
        if "user_attributes" in policy_elements:
            if isinstance(policy_elements["user_attributes"], list):
                for attr in policy_elements["user_attributes"]:
                    nodes.append({
                        "node_id": f"n{node_id_counter}",
                        "type": "user_attributes",
                        "content": attr
                    })
                    node_id_counter += 1
        
        # Add object_attributes
        if "object_attributes" in policy_elements:
            if isinstance(policy_elements["object_attributes"], list):
                for attr in policy_elements["object_attributes"]:
                    nodes.append({
                        "node_id": f"n{node_id_counter}",
                        "type": "object_attributes",
                        "content": attr
                    })
                    node_id_counter += 1
        
        # Add policy_classes
        if "policy_classes" in policy_elements:
            policy_class = policy_elements["policy_classes"]
            if isinstance(policy_class, str):
                nodes.append({
                    "node_id": f"n{node_id_counter}",
                    "type": "policy_classes",
                    "content": policy_class
                })
                node_id_counter += 1
    
    return {"nodes": nodes}


def convert_ground_truth_to_relation_classification_format(
    ground_truth_json: Dict,
    from_entity: str,
    to_entity: str,
    relation_type: str = "assign"
) -> Dict:
    """
    Convert ground truth JSON to relation classification output format.
    """
    exists = "No"
    explanation = f"No {relation_type} relationship found between {from_entity} and {to_entity}"
    subrelations = []
    
    # Check assignments
    if relation_type == "assign" and "assignments" in ground_truth_json:
        assignments_raw = ground_truth_json["assignments"]
        assignments_list = assignments_raw.values() if isinstance(assignments_raw, dict) else assignments_raw
        for assign_data in assignments_list:
            if assign_data.get("from") == from_entity and assign_data.get("to") == to_entity:
                exists = "Yes"
                explanation = f"Direct arrow from '{from_entity}' to '{to_entity}' indicating an 'assign' relationship."
                break
    
    # Check associations (permit)
    elif relation_type in ["permit", "associate"] and "associations" in ground_truth_json:
        associations_raw = ground_truth_json["associations"]
        associations_list = associations_raw.values() if isinstance(associations_raw, dict) else associations_raw
        for assoc_data in associations_list:
            if assoc_data.get("from") == from_entity and assoc_data.get("to") == to_entity:
                exists = "Yes"
                subrelations = assoc_data.get("weight", [])
                weight_str = ", ".join(subrelations) if subrelations else "permit"
                explanation = f"Direct arrow from '{from_entity}' to '{to_entity}' indicating a 'permit' relationship with weights: {weight_str}."
                break
    
    # Check prohibitions
    elif relation_type == "prohibit" and "prohibitions" in ground_truth_json:
        prohibitions_raw = ground_truth_json["prohibitions"]
        prohibitions_list = prohibitions_raw.values() if isinstance(prohibitions_raw, dict) else prohibitions_raw
        for prohib_data in prohibitions_list:
            if prohib_data.get("from") == from_entity and prohib_data.get("to") == to_entity:
                exists = "Yes"
                explanation = f"Direct arrow from '{from_entity}' to '{to_entity}' indicating a 'prohibit' relationship."
                break
    
    confidence = "high" if exists == "Yes" else "medium"
    
    return {
        "entity1": from_entity,
        "entity2": to_entity,
        "relation": relation_type,
        "exists": exists,
        "confidence": confidence,
        "explanation": explanation,
        "subrelations": subrelations
    }


def convert_ground_truth_to_path_enumeration_format(ground_truth_json: Dict) -> Dict:
    """
    Convert ground truth JSON to path enumeration output format.

    Creates nodes, edges, and paths from the ground truth data, including
    entity-to-node mapping and path finding between user attributes and policy classes.

    Args:
        ground_truth_json: Ground truth dictionary with policy_elements, assignments, associations, prohibitions

    Returns:
        Dictionary with nodes, edges, paths, and graph_metadata keys
    """
    # First, create node mapping (entity name -> node_id)
    entity_to_node_id = {}
    nodes = []
    node_id_counter = 1
    
    # Extract all entities and create nodes
    if "policy_elements" in ground_truth_json:
        policy_elements = ground_truth_json["policy_elements"]
        
        # Add user_attributes
        if "user_attributes" in policy_elements:
            if isinstance(policy_elements["user_attributes"], list):
                for attr in policy_elements["user_attributes"]:
                    node_id = f"n{node_id_counter}"
                    entity_to_node_id[attr] = node_id
                    nodes.append({
                        "node_id": node_id,
                        "type": "user_attributes",
                        "content": attr
                    })
                    node_id_counter += 1
        
        # Add object_attributes
        if "object_attributes" in policy_elements:
            if isinstance(policy_elements["object_attributes"], list):
                for attr in policy_elements["object_attributes"]:
                    node_id = f"n{node_id_counter}"
                    entity_to_node_id[attr] = node_id
                    nodes.append({
                        "node_id": node_id,
                        "type": "object_attributes",
                        "content": attr
                    })
                    node_id_counter += 1
        
        # Add policy_classes
        if "policy_classes" in policy_elements:
            policy_class = policy_elements["policy_classes"]
            if isinstance(policy_class, str):
                node_id = f"n{node_id_counter}"
                entity_to_node_id[policy_class] = node_id
                nodes.append({
                    "node_id": node_id,
                    "type": "policy_classes",
                    "content": policy_class
                })
                node_id_counter += 1
    
    # Create edges from assignments
    edges = []
    if "assignments" in ground_truth_json:
        assignments_raw = ground_truth_json["assignments"]
        assignments_list = assignments_raw.values() if isinstance(assignments_raw, dict) else assignments_raw
        for assign_data in assignments_list:
            from_entity = assign_data.get("from")
            to_entity = assign_data.get("to")
            if from_entity in entity_to_node_id and to_entity in entity_to_node_id:
                edges.append({
                    "from_id": entity_to_node_id[from_entity],
                    "source_name": from_entity,
                    "to_id": entity_to_node_id[to_entity],
                    "target_name": to_entity,
                    "relationship": "assign",
                    "subrelations": []
                })
    
    # Create edges from associations (permit)
    if "associations" in ground_truth_json:
        associations_raw = ground_truth_json["associations"]
        associations_list = associations_raw.values() if isinstance(associations_raw, dict) else associations_raw
        for assoc_data in associations_list:
            from_entity = assoc_data.get("from")
            to_entity = assoc_data.get("to")
            if from_entity in entity_to_node_id and to_entity in entity_to_node_id:
                edges.append({
                    "from_id": entity_to_node_id[from_entity],
                    "source_name": from_entity,
                    "to_id": entity_to_node_id[to_entity],
                    "target_name": to_entity,
                    "relationship": "permit",
                    "subrelations": assoc_data.get("weight", [])
                })
    
    # Create edges from prohibitions
    if "prohibitions" in ground_truth_json:
        prohibitions_raw = ground_truth_json["prohibitions"]
        prohibitions_list = prohibitions_raw.values() if isinstance(prohibitions_raw, dict) else prohibitions_raw
        for prohib_data in prohibitions_list:
            from_entity = prohib_data.get("from")
            to_entity = prohib_data.get("to")
            if from_entity in entity_to_node_id and to_entity in entity_to_node_id:
                edges.append({
                    "from_id": entity_to_node_id[from_entity],
                    "source_name": from_entity,
                    "to_id": entity_to_node_id[to_entity],
                    "target_name": to_entity,
                    "relationship": "prohibit",
                    "subrelations": []
                })
    
    # Generate paths by following edges (simple paths from user_attributes to policy_classes)
    paths = []
    path_id_counter = 1
    
    # Find all paths from user_attributes through to policy_classes
    policy_class_nodes = [n["node_id"] for n in nodes if n["type"] == "policy_classes"]
    user_attribute_nodes = [n["node_id"] for n in nodes if n["type"] == "user_attributes"]
    
    for user_node_id in user_attribute_nodes:
        for policy_node_id in policy_class_nodes:
            # Find path from user to policy
            path = find_path(user_node_id, policy_node_id, edges, entity_to_node_id)
            if path:
                relationships = []
                for i in range(len(path) - 1):
                    # Find edge between path[i] and path[i+1]
                    for edge in edges:
                        if edge["from_id"] == path[i] and edge["to_id"] == path[i+1]:
                            relationships.append(edge["relationship"])
                            break
                
                if relationships:
                    paths.append({
                        "path_id": f"path_{path_id_counter}",
                        "nodes": path,
                        "relationships": relationships,
                        "length": len(relationships),
                        "description": f"Path from {nodes[int(path[0][1:])-1]['content']} to {nodes[int(path[-1][1:])-1]['content']}"
                    })
                    path_id_counter += 1
    
    # If no paths found, create simple paths from direct edges
    if not paths and edges:
        for i, edge in enumerate(edges[:3]):  # Limit to first 3 for few-shot
            paths.append({
                "path_id": f"path_{i+1}",
                "nodes": [edge["from_id"], edge["to_id"]],
                "relationships": [edge["relationship"]],
                "length": 1,
                "description": f"Direct {edge['relationship']} from {edge['source_name']} to {edge['target_name']}"
            })
    
    graph_metadata = {
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "total_paths": len(paths)
    }
    
    return {
        "nodes": nodes,
        "edges": edges,
        "paths": paths,
        "graph_metadata": graph_metadata
    }


def find_path(start_node_id: str, end_node_id: str, edges: List[Dict], entity_to_node_id: Dict) -> Optional[List[str]]:
    """
    Find a path between two nodes using BFS (Breadth-First Search).

    Args:
        start_node_id: Starting node ID
        end_node_id: Ending node ID
        edges: List of edge dictionaries with 'from_id' and 'to_id' keys
        entity_to_node_id: Mapping of entity names to node IDs (unused in current implementation)

    Returns:
        List of node IDs representing the path from start to end, or None if no path exists
    """
    # Build adjacency list from edges
    adj = {}
    for edge in edges:
        from_id = edge["from_id"]
        to_id = edge["to_id"]
        if from_id not in adj:
            adj[from_id] = []
        adj[from_id].append(to_id)

    # BFS to find path
    queue = deque([(start_node_id, [start_node_id])])
    visited = {start_node_id}

    while queue:
        current, path = queue.popleft()

        if current == end_node_id:
            return path

        if current in adj:
            for neighbor in adj[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

    return None


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode image file to base64 string.

    First tries direct binary reading, then falls back to PIL for format conversion
    if the direct approach fails.

    Args:
        image_path: Path to the image file

    Returns:
        Base64 encoded string representation of the image

    Raises:
        Exception: If both encoding methods fail
    """
    try:
        # Try direct binary reading first
        with open(image_path, "rb") as image_file:
            data = image_file.read()
            return base64.b64encode(data).decode('utf-8')
    except Exception:
        # Fallback to PIL with truncated image support
        try:
            from PIL import Image, ImageFile
            from io import BytesIO
            ImageFile.LOAD_TRUNCATED_IMAGES = True

            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                buf = BytesIO()
                img.save(buf, format="PNG")
                return base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as inner_e:
            raise Exception(f"Failed to encode image {image_path}: {inner_e}")


def create_vision_message(text_content: str, image_base64: str) -> Dict[str, Any]:
    """
    Create a vision message with text and image content.

    Args:
        text_content: The text prompt to include
        image_base64: Base64 encoded image data

    Returns:
        Message dictionary with text and image content
    """
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text_content},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}", "detail": "high"}}
        ]
    }


def build_message_sequence(prompt_text: str, image_base64: str, few_shot_examples: Optional[List[Dict]] = None) -> List[Dict]:
    """
    Build a complete message sequence for vision tasks.

    Args:
        prompt_text: The main task prompt
        image_base64: Base64 encoded image
        few_shot_examples: Optional few-shot examples to include

    Returns:
        Complete message sequence including system prompt, examples, and task
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT_VISION}]

    # Add few-shot examples if provided
    if few_shot_examples:
        messages.extend(few_shot_examples)

    # Add the actual task
    messages.append(create_vision_message(prompt_text, image_base64))

    return messages


def generate_negative_examples(positives: List[Dict], entities: set, max_per_type: int = 10) -> List[Dict]:
    """
    Generate negative examples for each relation type that has positive examples.

    Args:
        positives: List of positive triple dictionaries
        entities: Set of all entity names
        max_per_type: Maximum negative examples per relation type

    Returns:
        List of negative triple dictionaries
    """
    # Get relation types that have positive examples
    relation_types_with_positives = set(p["relation"] for p in positives)
    pos_pairs = {(p["from_entity"], p["to_entity"]) for p in positives}

    negatives = []
    ent_list = list(entities)

    for relation_type in relation_types_with_positives:
        negatives_for_type = 0
        for from_entity in ent_list:
            for to_entity in ent_list:
                if from_entity != to_entity and (from_entity, to_entity) not in pos_pairs:
                    negatives.append({
                        "from_entity": from_entity,
                        "to_entity": to_entity,
                        "relation": relation_type
                    })
                    negatives_for_type += 1
                    if negatives_for_type >= max_per_type:
                        break
            if negatives_for_type >= max_per_type:
                break

    return negatives


def select_representative_examples(positives: List[Dict], negatives: List[Dict], max_types: int = 2) -> Tuple[List[Dict], List[Dict]]:
    """
    Select representative positive and negative examples with relation type balance.

    Args:
        positives: List of positive triple dictionaries
        negatives: List of negative triple dictionaries
        max_types: Maximum number of relation types to include

    Returns:
        Tuple of (selected_positives, selected_negatives)
    """
    # Group by relation type
    pos_by_type = {}
    neg_by_type = {}

    for p in positives:
        rel_type = p["relation"]
        pos_by_type.setdefault(rel_type, []).append(p)

    for n in negatives:
        rel_type = n["relation"]
        neg_by_type.setdefault(rel_type, []).append(n)

    # Select examples from relation types (sorted for determinism)
    selected_pos = []
    selected_neg = []

    for rel_type in sorted(pos_by_type.keys())[:max_types]:
        if pos_by_type[rel_type]:
            selected_pos.append(pos_by_type[rel_type][0])  # First positive
        if rel_type in neg_by_type and neg_by_type[rel_type]:
            selected_neg.append(neg_by_type[rel_type][0])  # First negative

    return selected_pos, selected_neg


def load_ground_truth(json_path: str) -> Dict:
    """
    Load ground truth data from JSON file.

    Args:
        json_path: Path to the ground truth JSON file

    Returns:
        Parsed JSON data as dictionary
    """
    with open(json_path, 'r') as f:
        return json.load(f)


def extract_positive_triples(ground_truth: Dict) -> List[Dict[str, str]]:
    """
    Extract all positive relationship triples from ground truth data.

    Args:
        ground_truth: Ground truth dictionary with assignments, associations, prohibitions

    Returns:
        List of triple dictionaries with 'from_entity', 'to_entity', 'relation' keys
    """
    positives = []

    # Extract assignments
    if "assignments" in ground_truth:
        assignments_raw = ground_truth["assignments"]
        assignments_list = assignments_raw.values() if isinstance(assignments_raw, dict) else assignments_raw
        for item in assignments_list:
            positives.append({
                "from_entity": item.get("from"),
                "to_entity": item.get("to"),
                "relation": "assign"
            })

    # Extract associations (permit relationships)
    if "associations" in ground_truth:
        associations_raw = ground_truth["associations"]
        associations_list = associations_raw.values() if isinstance(associations_raw, dict) else associations_raw
        for item in associations_list:
            positives.append({
                "from_entity": item.get("from"),
                "to_entity": item.get("to"),
                "relation": "permit"
            })

    # Extract prohibitions
    if "prohibitions" in ground_truth:
        prohibitions_raw = ground_truth["prohibitions"]
        prohibitions_list = prohibitions_raw.values() if isinstance(prohibitions_raw, dict) else prohibitions_raw
        for item in prohibitions_list:
            positives.append({
                "from_entity": item.get("from"),
                "to_entity": item.get("to"),
                "relation": "prohibit"
            })

    return positives


def extract_all_entities(ground_truth: Dict) -> set:
    """
    Extract all unique entity names from ground truth data.

    Args:
        ground_truth: Ground truth dictionary

    Returns:
        Set of all entity names
    """
    entities = set()

    if "policy_elements" in ground_truth:
        pe = ground_truth["policy_elements"]
        for key in ["user_attributes", "object_attributes"]:
            attrs = pe.get(key, [])
            if isinstance(attrs, list):
                entities.update(attrs)

        policy_class = pe.get("policy_classes")
        if isinstance(policy_class, str):
            entities.add(policy_class)
        elif isinstance(policy_class, list):
            entities.update(policy_class)

    return entities


def create_message_pair(prompt_template: str, image_base64: str, triple_data: Dict, expected_output: str) -> List[Dict]:
    """
    Create a user-assistant message pair for few-shot examples.

    Args:
        prompt_template: The prompt template to format
        image_base64: Base64 encoded image
        triple_data: Triple data for prompt formatting
        expected_output: Expected JSON output string

    Returns:
        List containing user message and assistant response
    """
    if "from_entity" in triple_data and "to_entity" in triple_data:
        # Relation classification prompt
        prompt = prompt_template.format(
            from_entity=triple_data["from_entity"],
            to_entity=triple_data["to_entity"],
            relation_type=triple_data["relation"]
        )
    else:
        # Entity extraction or path enumeration prompt
        prompt = prompt_template

    return [
        create_vision_message(prompt, image_base64),
        {"role": "assistant", "content": expected_output}
    ]

# ============================================================================
# FEW-SHOT EXAMPLE GENERATION FUNCTIONS
# ============================================================================


def generate_few_shot_examples_for_entity_extraction(
    image_without_labels_path: str = FEW_SHOT_IMAGE_PATH,
    image_with_labels_path: str = FEW_SHOT_IMAGE_B_PATH,
    ground_truth_json_path: str = FEW_SHOT_JSON_PATH
) -> List[Dict]:
    """
    Generate two-shot examples for entity extraction using multiple conversation turns.

    Uses the same ground truth output for both images to demonstrate consistency.

    Args:
        image_without_labels_path: Path to image without label overlays
        image_with_labels_path: Path to image with label overlays
        ground_truth_json_path: Path to ground truth JSON file

    Returns:
        List of message dictionaries for few-shot learning
    """
    # Load and convert ground truth
    ground_truth = load_ground_truth(ground_truth_json_path)
    expected_output = convert_ground_truth_to_entity_extraction_format(ground_truth)
    expected_output_json = json.dumps(expected_output, indent=2)

    # Encode images
    image_without_labels_base64 = encode_image_to_base64(image_without_labels_path)
    image_with_labels_base64 = encode_image_to_base64(image_with_labels_path)

    # Create message pairs for each image
    messages = []
    messages.extend(create_message_pair(
        ENTITY_EXTRACTION_PROMPT, image_without_labels_base64, {}, expected_output_json
    ))
    messages.extend(create_message_pair(
        ENTITY_EXTRACTION_PROMPT, image_with_labels_base64, {}, expected_output_json
    ))

    return messages


def generate_few_shot_examples_for_relation_classification(
    image_without_labels_path: str = FEW_SHOT_IMAGE_PATH,
    image_with_labels_path: str = FEW_SHOT_IMAGE_B_PATH,
    ground_truth_json_path: str = FEW_SHOT_JSON_PATH,
    example_relations: Optional[List[Dict]] = None
) -> List[Dict]:
    """
    Generate four-shot examples for relation classification (positive and negative examples per image).

    Uses deterministic selection of representative examples for reproducibility.
    Ensures negative examples match the relation types of positive examples.

    Args:
        image_without_labels_path: Path to image without label overlays
        image_with_labels_path: Path to image with label overlays
        ground_truth_json_path: Path to ground truth JSON file
        example_relations: Optional pre-selected relations (unused in current implementation)

    Returns:
        List of message dictionaries for few-shot learning
    """
    # Load ground truth and extract data
    ground_truth = load_ground_truth(ground_truth_json_path)
    positives = extract_positive_triples(ground_truth)
    entities = extract_all_entities(ground_truth)

    if not positives:
        raise ValueError("No positive examples found in ground truth")

    # Generate and select examples
    negatives = generate_negative_examples(positives, entities)
    selected_pos, selected_neg = select_representative_examples(positives, negatives)

    # Fallback handling
    if not selected_pos:
        selected_pos = [positives[0]]
    if not selected_neg:
        # Create fallback negative with same relation type as first positive
        first_pos = selected_pos[0]
        fallback_neg = {
            "from_entity": list(entities)[0] if entities else "unknown_entity",
            "to_entity": list(entities)[1] if len(entities) > 1 else "unknown_entity",
            "relation": first_pos["relation"]
        }
        selected_neg = [fallback_neg]

    # Encode images
    image_without_labels_base64 = encode_image_to_base64(image_without_labels_path)
    image_with_labels_base64 = encode_image_to_base64(image_with_labels_path)

    # Create message scenarios: (image, relation_data) pairs
    scenarios = [
        (image_without_labels_base64, selected_pos[0]),
        (image_without_labels_base64, selected_neg[0]),
        (image_with_labels_base64, selected_pos[1] if len(selected_pos) > 1 else selected_pos[0]),
        (image_with_labels_base64, selected_neg[1] if len(selected_neg) > 1 else selected_neg[0])
    ]

    # Build message sequence
    messages = []
    for img_base64, rel_data in scenarios:
        expected_output = convert_ground_truth_to_relation_classification_format(
            ground_truth,
            rel_data["from_entity"],
            rel_data["to_entity"],
            rel_data["relation"]
        )
        expected_output_json = json.dumps(expected_output, indent=2)

        messages.extend(create_message_pair(
            RELATION_CLASSIFICATION_PROMPT, img_base64, rel_data, expected_output_json
        ))

    return messages


def generate_few_shot_examples_for_path_enumeration(
    image_without_labels_path: str = FEW_SHOT_IMAGE_PATH,
    image_with_labels_path: str = FEW_SHOT_IMAGE_B_PATH,
    ground_truth_json_path: str = FEW_SHOT_JSON_PATH
) -> List[Dict]:
    """
    Generate two-shot examples for path enumeration using multiple conversation turns.

    Uses the same ground truth output for both images to demonstrate consistency.

    Args:
        image_without_labels_path: Path to image without label overlays
        image_with_labels_path: Path to image with label overlays
        ground_truth_json_path: Path to ground truth JSON file

    Returns:
        List of message dictionaries for few-shot learning
    """
    # Load and convert ground truth
    ground_truth = load_ground_truth(ground_truth_json_path)
    expected_output = convert_ground_truth_to_path_enumeration_format(ground_truth)
    expected_output_json = json.dumps(expected_output, indent=2)

    # Encode images
    image_without_labels_base64 = encode_image_to_base64(image_without_labels_path)
    image_with_labels_base64 = encode_image_to_base64(image_with_labels_path)

    # Create message pairs for each image
    messages = []
    messages.extend(create_message_pair(
        PATH_ENUMERATION_PROMPT, image_without_labels_base64, {}, expected_output_json
    ))
    messages.extend(create_message_pair(
        PATH_ENUMERATION_PROMPT, image_with_labels_base64, {}, expected_output_json
    ))

    return messages

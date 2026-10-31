"""
Tool registry: the single place that defines each tool's name, description, JSON
schema and handler. The orchestrator, the router prompt, the README and the tests
all read from here, so documentation cannot drift from behaviour.

Privileged context (the session, the database) is injected by the orchestrator via
ToolContext. The LLM can never supply it, so it cannot claim an order is "verified".
"""
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from app.tools.catalogue import search_products
from app.tools.crm import customer_profile
from app.tools.db import CATEGORIES
from app.tools.orders import lookup_order
from app.tools.returns import check_return_eligibility
from app.tools.types import ToolContext, ToolResult

Handler = Callable[[dict, ToolContext], Awaitable[ToolResult]]


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict                 # JSON schema shown to the LLM
    required: list[str]
    handler: Handler
    questions: dict = field(default_factory=dict)  # missing-field -> question to ask the user


TOOLS: dict[str, Tool] = {
    "search_products": Tool(
        name="search_products",
        description="Search BrightCart's product catalogue by keywords, category, maximum "
                    "price in rupees, and stock availability. Returns up to 3 matches with prices.",
        parameters={"type": "object", "properties": {
            "query": {"type": "string", "description": "product keywords, e.g. 'wireless headphones'"},
            "category": {"type": "string", "enum": CATEGORIES},
            "max_price": {"type": "number", "description": "maximum price in rupees"},
            "in_stock_only": {"type": "boolean"}}},
        required=[],  # handler requires at least one of query / category / max_price
        handler=search_products,
        questions={"query": "What kind of product are you looking for?"},
    ),
    "lookup_order": Tool(
        name="lookup_order",
        description="Look up an order's status, carrier, tracking reference and estimated "
                    "delivery. Requires the order number and the customer's verification code "
                    "(last four digits of the phone number on the order).",
        parameters={"type": "object", "properties": {
            "order_id": {"type": "string", "description": "e.g. BC1024"},
            "verification_code": {"type": "string",
                                  "description": "last 4 digits of the phone number on the order"}},
        },
        required=["order_id", "verification_code"],
        handler=lookup_order,
        questions={"order_id": "Sure. What is your order number? It starts with B C.",
                   "verification_code": "For security, what are the last four digits of the "
                                        "phone number on the order?"},
    ),
    "check_return_eligibility": Tool(
        name="check_return_eligibility",
        description="Check whether an item from a delivered order can be returned under "
                    "BrightCart's return policy (30 days, unused, original packaging).",
        parameters={"type": "object", "properties": {
            "order_id": {"type": "string"},
            "item": {"type": "string", "description": "product name or id, e.g. 'headphones'"},
            "condition": {"type": "string", "enum": ["unused", "used", "damaged"]},
            "has_original_packaging": {"type": "boolean"}}},
        required=["order_id", "item", "condition", "has_original_packaging"],
        handler=check_return_eligibility,
        questions={"order_id": "I can check that. What is your order number?",
                   "item": "Which item from the order would you like to return?",
                   "condition": "Is the item unused, used, or damaged?",
                   "has_original_packaging": "Do you still have the original packaging?",
                   "verification_code": "For security, what are the last four digits of the "
                                        "phone number on the order?"},
    ),
    "customer_profile": Tool(
        name="customer_profile",
        description="Remember or recall a returning customer's details across chats: save their name, "
                    "contact number or email, or a stated preference; read back what is saved; or "
                    "delete it when they ask to be forgotten.",
        parameters={"type": "object", "properties": {
            "action": {"type": "string", "enum": ["get", "update", "forget"]},
            "name": {"type": "string"},
            "contact": {"type": "string", "description": "email address or phone number"},
            "preference": {"type": "string", "description": "e.g. 'prefers black headphones'"}}},
        required=[],
        handler=customer_profile,
        questions={"preference": "What would you like me to remember?"},
    ),
}


def tool_schemas() -> list[dict]:
    """OpenAI/Ollama-style function schemas (used in docs and tests)."""
    return [{"name": t.name, "description": t.description, "parameters": {
        **t.parameters, "required": t.required}} for t in TOOLS.values()]

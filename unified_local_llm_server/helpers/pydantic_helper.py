from __future__ import annotations

from typing import Any, Dict, List, Optional, Type, Literal

from pydantic import BaseModel, RootModel, create_model


def dict_to_pydantic_schema(
    schema_dict: Dict[str, Any],
    name_for_llm: str = "DynamicSchema",
    *,
    as_kv_list: bool = False,
) -> Type[BaseModel]:
    if as_kv_list:
        value_types = []
        for v in schema_dict.values():
            value_types.append(v if isinstance(v, type) else type(v))
        value_type = value_types[0] if value_types and all(t is value_types[0] for t in value_types) else Any
        item_fields = {
            "key": (str, ...),
            "value": (value_type, ...),
        }
        item_model = create_model(f"{name_for_llm}Item", **item_fields)
        return type(name_for_llm, (RootModel[list[item_model]],), {})

    fields = {}
    for k, v in schema_dict.items():
        if isinstance(v, list):
            if not v:
                fields[k] = (List[Any], ...)
            elif len(v) == 1:
                item = v[0]
                if isinstance(item, type):
                    fields[k] = (List[item], ...)
                elif isinstance(item, dict):
                    inner_model = dict_to_pydantic_schema(item, name_for_llm=f"{k.capitalize()}Item")
                    fields[k] = (List[inner_model], ...)
                else:
                    literal_type = Literal[item]
                    fields[k] = (List[literal_type], ...)
            elif all(isinstance(item, (str, int, float, bool)) for item in v):
                literal_type = Literal[tuple(v)]
                fields[k] = (List[literal_type], ...)
            else:
                raise ValueError(f"Ambiguous or unsupported list field '{k}': {v}")
        elif isinstance(v, dict):
            inner_model = dict_to_pydantic_schema(v, name_for_llm=f"{k.capitalize()}Model")
            fields[k] = (inner_model, ...)
        elif v is None:
            fields[k] = (Optional[Any], None)
        elif isinstance(v, type):
            fields[k] = (v, ...)
        else:
            fields[k] = (type(v), v)
    return create_model(name_for_llm, **fields)


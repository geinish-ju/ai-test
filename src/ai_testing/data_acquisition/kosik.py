from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

from ai_testing.data_acquisition.common import (
    JsonApiClient,
    JsonObject,
    Record,
    compact_text_list,
    date_part,
    extract_number,
    first_value,
    int_value,
    mapping_list_value,
    mapping_value,
    materialize_mapping_list,
    require_json_object,
    text_value,
)

ORDER_LIST_PATH = "/api/front/profile/order-list"
ORDER_DETAIL_PATH_TEMPLATE = "/api/front/profile/order/{order_id}"
PRODUCT_BY_SLUG_PATH_TEMPLATE = "/api/front/product/slug/{slug}"

ProductDetailLoader = Callable[[JsonObject], JsonObject | None]

KOSIK_FIELD_SOURCES = {
    "order_id": "order_detail.id",
    "order_number": "order_detail.number",
    "order_created_at": "order_detail.created",
    "delivery_slot_start": "order_detail.transport.start",
    "delivery_slot_end": "order_detail.transport.end",
    "order_item_id": "order_detail.orderProducts[].id",
    "product_id": "order_detail.orderProducts[].product.id",
    "product_slug": "order_detail.orderProducts[].product.url",
    "product_name": "order_detail.orderProducts[].product.name",
    "brand": "order_detail.orderProducts[].product.brand or product_detail.product.detail.brand",
    "category": "product_detail.categoryTree[-1].name or order_product.product.mainCategory.name",
    "category_id": "product_detail.categoryTree[-1].id or order_product.product.mainCategory.id",
    "category_path": "product_detail.categoryTree[].name or product_detail.breadcrumbs[].name",
    "quantity": "order_detail.orderProducts[].deliveredQuantity",
    "quantity_ordered": "order_detail.orderProducts[].orderedQuantity",
    "quantity_delivered": "order_detail.orderProducts[].deliveredQuantity",
    "unit": "order_detail.orderProducts[].product.unit",
    "price_total": "order_detail.orderProducts[].price",
    "price_unit": "order_detail.orderProducts[].product.price",
    "price_per_unit": "order_detail.orderProducts[].product.pricePerUnit.price",
    "currency": "CZK",
    "product_description": "product_detail.product.detail.description[].value",
    "ingredients_text": "product_detail.product.detail.ingredients[].value",
}


@dataclass(frozen=True)
class KosikApiClient:
    base_url: str = "https://www.kosik.cz"
    cookies: Mapping[str, str] | str | None = None
    headers: Mapping[str, str] | None = None
    timeout_seconds: float = 30.0
    order_list_path: str = ORDER_LIST_PATH
    order_detail_path_template: str = ORDER_DETAIL_PATH_TEMPLATE
    product_by_slug_path_template: str = PRODUCT_BY_SLUG_PATH_TEMPLATE

    def get_orders(
        self,
        page_limit: int = 50,
        include_archived: bool = True,
    ) -> list[JsonObject]:
        if page_limit <= 0:
            raise ValueError("Kosik order page limit must be greater than zero")

        orders: list[JsonObject] = []
        offset = 0
        while True:
            page_payload = self.get_order_list_page(
                offset=offset,
                limit=page_limit,
                include_archived=include_archived,
            )
            page = mapping_list_value(page_payload, "orders")
            if not page:
                break

            orders.extend(page)
            if len(page) < page_limit:
                break

            offset += len(page)

        return orders

    def get_order_list_page(
        self,
        offset: int,
        limit: int,
        include_archived: bool,
    ) -> JsonObject:
        query = urlencode(
            {
                "limit": limit,
                "showArchived": str(include_archived).lower(),
                "offset": offset,
            }
        )
        client = self._client()
        return require_json_object(client.get_json(f"{self.order_list_path}?{query}"), "Kosik")

    def get_order_detail(self, order_id: str | int) -> JsonObject:
        client = self._client()
        path = self.order_detail_path_template.format(order_id=order_id)
        return require_json_object(client.get_json(path), "Kosik")

    def get_product_by_slug(self, slug: str) -> JsonObject:
        client = self._client()
        path = self.product_by_slug_path_template.format(slug=slug)
        return require_json_object(client.get_json(path), "Kosik")

    def _client(self) -> JsonApiClient:
        return JsonApiClient(
            base_url=self.base_url,
            service_name="Kosik",
            cookies=self.cookies,
            headers=self.headers,
            timeout_seconds=self.timeout_seconds,
        )


@dataclass(frozen=True)
class KosikOrderHistoryAdapter:
    client: KosikApiClient
    include_raw: bool = False
    enrich_products: bool = True
    order_page_limit: int = 50
    include_archived_orders: bool = True

    def fetch_order_item_records(self) -> list[Record]:
        orders = self.client.get_orders(
            page_limit=self.order_page_limit,
            include_archived=self.include_archived_orders,
        )
        product_details_by_slug: dict[str, JsonObject | None] = {}

        def load_product_detail(product: JsonObject) -> JsonObject | None:
            if not self.enrich_products:
                return None

            slug = product_slug(product)
            if slug is None:
                return None

            if slug not in product_details_by_slug:
                try:
                    product_details_by_slug[slug] = self.client.get_product_by_slug(slug)
                except (ConnectionError, PermissionError, TypeError, ValueError):
                    product_details_by_slug[slug] = None
            return product_details_by_slug[slug]

        records: list[Record] = []
        for order in orders:
            order_id = _extract_order_id(order)
            if order_id is None:
                raise ValueError("Cannot extract order identifier from Kosik order-list item")

            order_detail = self.client.get_order_detail(order_id)
            records.extend(
                normalize_order_detail(
                    order_detail,
                    fallback_order_id=order_id,
                    fallback_order_created_at=text_value(order.get("created")),
                    include_raw=self.include_raw,
                    product_detail_loader=load_product_detail,
                )
            )
        return records


def normalize_order_detail(
    order_detail: JsonObject,
    fallback_order_id: str | int | None = None,
    fallback_order_created_at: str | None = None,
    include_raw: bool = False,
    product_detail_loader: ProductDetailLoader | None = None,
) -> list[Record]:
    order_id = _extract_order_id(order_detail) or fallback_order_id
    if order_id is None:
        raise ValueError("Cannot extract order identifier from Kosik order detail")

    return [
        _normalize_order_product(
            order_detail=order_detail,
            order_id=order_id,
            fallback_order_created_at=fallback_order_created_at,
            item=item,
            include_raw=include_raw,
            product_detail_loader=product_detail_loader,
        )
        for item in mapping_list_value(order_detail, "orderProducts")
    ]


def _normalize_order_product(
    order_detail: JsonObject,
    order_id: str | int,
    fallback_order_created_at: str | None,
    item: JsonObject,
    include_raw: bool,
    product_detail_loader: ProductDetailLoader | None,
) -> Record:
    product = _product_from_order_item(item)
    product_detail = product_detail_loader(product) if product_detail_loader is not None else None
    product_detail_product = mapping_value(product_detail or {}, "product")
    product_detail_inner = mapping_value(product_detail_product or {}, "detail")
    transport = mapping_value(order_detail, "transport")
    categories = _category_nodes(product, product_detail)
    main_category = categories[0] if categories else None
    leaf_category = categories[-1] if categories else None
    price_per_unit = mapping_value(product, "pricePerUnit")
    package_quantity = mapping_value(product, "productQuantity")
    order_created_at = text_value(order_detail.get("created")) or fallback_order_created_at
    quantity_delivered = extract_number(item.get("deliveredQuantity"))

    record: Record = {
        "shop": "kosik",
        "order_id": order_id,
        "order_number": text_value(order_detail.get("number")),
        "order_date": date_part(order_created_at),
        "order_created_at": order_created_at,
        "delivery_slot_start": text_value((transport or {}).get("start")),
        "delivery_slot_end": text_value((transport or {}).get("end")),
        "order_item_id": item.get("id"),
        "product_id": product.get("id"),
        "product_slug": product_slug(product),
        "product_url": text_value(product.get("url")),
        "product_name": text_value(product.get("name")),
        "brand": _brand(product, product_detail_inner),
        "main_category_id": _node_id(main_category),
        "main_category": _node_name(main_category),
        "category_id": _node_id(leaf_category),
        "category": _node_name(leaf_category),
        "category_path": _category_path(categories),
        "category_ids": [_node_id(node) for node in categories if _node_id(node) is not None],
        "quantity": quantity_delivered,
        "quantity_ordered": extract_number(item.get("orderedQuantity")),
        "quantity_delivered": quantity_delivered,
        "unit": text_value(product.get("unit")),
        "package_quantity": extract_number((package_quantity or {}).get("value")),
        "package_unit": text_value((package_quantity or {}).get("unit")),
        "price_total": extract_number(item.get("price")),
        "price_unit": extract_number(product.get("price")),
        "price_per_unit": extract_number((price_per_unit or {}).get("price")),
        "price_per_unit_unit": text_value((price_per_unit or {}).get("unit")),
        "currency": "CZK",
        "product_description": _detail_section_text(product_detail_inner, "description"),
        "ingredients_text": _detail_section_text(product_detail_inner, "ingredients"),
        "product_enriched": product_detail is not None,
    }

    if include_raw:
        record["raw"] = dict(item)

    return record


def _product_from_order_item(item: JsonObject) -> JsonObject:
    product = mapping_value(item, "product")
    if product is None:
        raise ValueError("Kosik orderProducts[] item does not contain product object")
    return product


def _extract_order_id(payload: JsonObject) -> str | int | None:
    value = first_value(payload, ("id", "orderId", "order_id", "uuid", "code"))
    return value if isinstance(value, (str, int)) else None


def product_slug(product: JsonObject) -> str | None:
    slug = text_value(product.get("slug"))
    if slug is not None:
        return slug

    url = text_value(product.get("url"))
    if url is None:
        return None
    return url.strip("/").split("/")[-1] or None


def _brand(product: JsonObject, product_detail: JsonObject | None) -> str | None:
    return (
        text_value(product.get("brand"))
        or text_value((product_detail or {}).get("brand"))
        or text_value(product.get("producer"))
    )


def _category_nodes(product: JsonObject, product_detail: JsonObject | None) -> list[JsonObject]:
    detail = product_detail or {}
    for key in ("categoryTree", "breadcrumbs"):
        nodes = _named_nodes(materialize_mapping_list(detail.get(key)))
        if nodes:
            return nodes

    detail_product = mapping_value(detail, "product")
    if detail_product is not None:
        detail_main_category = mapping_value(detail_product, "mainCategory")
        if detail_main_category is not None and _node_name(detail_main_category) is not None:
            return [detail_main_category]

    main_category = mapping_value(product, "mainCategory")
    if main_category is not None and _node_name(main_category) is not None:
        return [main_category]

    return []


def _named_nodes(nodes: list[JsonObject] | None) -> list[JsonObject]:
    if nodes is None:
        return []
    return [node for node in nodes if _node_name(node) is not None]


def _node_id(node: JsonObject | None) -> Any:
    if node is None:
        return None
    return int_value(node.get("id")) or text_value(node.get("id")) or text_value(node.get("url"))


def _node_name(node: JsonObject | None) -> str | None:
    if node is None:
        return None
    return text_value(node.get("name"))


def _category_path(categories: list[JsonObject]) -> list[str]:
    return compact_text_list([_node_name(node) for node in categories])


def _detail_section_text(product_detail: JsonObject | None, key: str) -> str | None:
    if product_detail is None:
        return None

    sections = materialize_mapping_list(product_detail.get(key)) or []
    values = compact_text_list([section.get("value") for section in sections])
    return "\n".join(values) if values else None

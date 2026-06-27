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
    money_amount,
    money_currency,
    require_json_list,
    require_json_object,
    text_value,
)

DELIVERED_ORDERS_PATH = "/api/v3/orders/delivered"
ORDER_DETAIL_PATH_TEMPLATE = "/api/v3/orders/{order_id}"
PRODUCT_CARD_PATH_TEMPLATE = "/api/v1/products/{product_id}/card"
PRODUCT_DETAIL_PATH_TEMPLATE = "/api/v1/products/{product_id}/detail"
PRODUCT_DETAIL_CONTENT_PATH_TEMPLATE = "/api/v1/products/{product_id}/detail/content"

ROHLIK_FIELD_SOURCES = {
    "order_id": "order_detail.id",
    "order_created_at": "order_detail.orderTime",
    "delivery_slot_start": "order_detail.deliverySlot.since",
    "delivery_slot_end": "order_detail.deliverySlot.till",
    "order_item_id": "order_detail.items[].orderFieldId",
    "product_id": "order_detail.items[].id",
    "product_slug": "product_card.slug",
    "product_name": "order_detail.items[].name or product_card.name",
    "brand": (
        "product_card.brand or product_detail.header.brandFilter or product_detail.filters[znacka]"
    ),
    "category": "product_detail.breadcrumbs[-1].name",
    "category_id": "product_detail.breadcrumbs[-1].id",
    "category_path": "product_detail.breadcrumbs[].name",
    "quantity": "order_detail.items[].amount",
    "unit": "order_detail.items[].unit",
    "package_unit": "order_detail.items[].textualAmount",
    "price_total": "order_detail.items[].priceComposition.total.amount",
    "price_unit": "order_detail.items[].priceComposition.unit.amount",
    "currency": "order_detail.items[].priceComposition.total.currency",
    "product_description": "product_detail_content.description",
    "ingredients_text": (
        "product_detail_content.composition.plainIngredients or "
        "product_detail_content.composition.ingredients[].title"
    ),
}


@dataclass(frozen=True)
class RohlikProductMetadata:
    card: JsonObject | None = None
    detail: JsonObject | None = None
    content: JsonObject | None = None


ProductMetadataLoader = Callable[[int], RohlikProductMetadata]


@dataclass(frozen=True)
class RohlikApiClient:
    base_url: str = "https://www.rohlik.cz"
    cookies: Mapping[str, str] | str | None = None
    headers: Mapping[str, str] | None = None
    timeout_seconds: float = 30.0
    delivered_orders_path: str = DELIVERED_ORDERS_PATH
    order_detail_path_template: str = ORDER_DETAIL_PATH_TEMPLATE
    product_card_path_template: str = PRODUCT_CARD_PATH_TEMPLATE
    product_detail_path_template: str = PRODUCT_DETAIL_PATH_TEMPLATE
    product_detail_content_path_template: str = PRODUCT_DETAIL_CONTENT_PATH_TEMPLATE

    def get_delivered_orders(
        self,
        page_limit: int = 10,
        include_archived: bool = True,
    ) -> list[JsonObject]:
        if page_limit <= 0:
            raise ValueError("Rohlik delivered orders page limit must be greater than zero")

        orders: list[JsonObject] = []
        offset = 0
        while True:
            page = self.get_delivered_orders_page(
                offset=offset,
                limit=page_limit,
                include_archived=include_archived,
            )
            if not page:
                break

            orders.extend(page)
            if len(page) < page_limit:
                break

            offset += len(page)

        return orders

    def get_delivered_orders_page(
        self,
        offset: int,
        limit: int,
        include_archived: bool,
    ) -> list[JsonObject]:
        query = urlencode(
            {
                "offset": offset,
                "limit": limit,
                "showArchived": str(include_archived).lower(),
            }
        )
        payload = self._client().get_json(f"{self.delivered_orders_path}?{query}")
        if isinstance(payload, Mapping):
            return mapping_list_value(payload, "orders")
        return require_json_list(payload, "Rohlik delivered orders")

    def get_order_detail(self, order_id: str | int) -> JsonObject:
        path = self.order_detail_path_template.format(order_id=order_id)
        return require_json_object(self._client().get_json(path), "Rohlik")

    def get_product_card(self, product_id: str | int) -> JsonObject:
        path = self.product_card_path_template.format(product_id=product_id)
        return require_json_object(self._client().get_json(path), "Rohlik product card")

    def get_product_detail(self, product_id: str | int) -> JsonObject:
        path = self.product_detail_path_template.format(product_id=product_id)
        return require_json_object(self._client().get_json(path), "Rohlik product detail")

    def get_product_detail_content(self, product_id: str | int) -> JsonObject:
        path = self.product_detail_content_path_template.format(product_id=product_id)
        return require_json_object(self._client().get_json(path), "Rohlik product detail content")

    def _client(self) -> JsonApiClient:
        return JsonApiClient(
            base_url=self.base_url,
            service_name="Rohlik",
            cookies=self.cookies,
            headers=self.headers,
            timeout_seconds=self.timeout_seconds,
        )


@dataclass(frozen=True)
class RohlikOrderHistoryAdapter:
    client: RohlikApiClient
    include_raw: bool = False
    enrich_products: bool = True
    include_product_content: bool = False
    order_page_limit: int = 10
    include_archived_orders: bool = True

    def fetch_order_item_records(self) -> list[Record]:
        orders = self.client.get_delivered_orders(
            page_limit=self.order_page_limit,
            include_archived=self.include_archived_orders,
        )
        product_metadata_by_id: dict[int, RohlikProductMetadata] = {}

        def load_product_metadata(product_id: int) -> RohlikProductMetadata:
            if not self.enrich_products:
                return RohlikProductMetadata()

            if product_id not in product_metadata_by_id:
                product_metadata_by_id[product_id] = RohlikProductMetadata(
                    card=_safe_get_json(lambda: self.client.get_product_card(product_id)),
                    detail=_safe_get_json(lambda: self.client.get_product_detail(product_id)),
                    content=(
                        _safe_get_json(lambda: self.client.get_product_detail_content(product_id))
                        if self.include_product_content
                        else None
                    ),
                )
            return product_metadata_by_id[product_id]

        records: list[Record] = []
        for order in orders:
            order_id = _extract_order_id(order)
            if order_id is None:
                raise ValueError("Cannot extract order identifier from Rohlik delivered order")

            order_detail = self.client.get_order_detail(order_id)
            records.extend(
                normalize_order_detail(
                    order_detail,
                    fallback_order_id=order_id,
                    fallback_order_created_at=text_value(order.get("orderTime")),
                    include_raw=self.include_raw,
                    product_metadata_loader=load_product_metadata,
                )
            )
        return records


def normalize_order_detail(
    order_detail: JsonObject,
    fallback_order_id: str | int | None = None,
    fallback_order_created_at: str | None = None,
    include_raw: bool = False,
    product_metadata_loader: ProductMetadataLoader | None = None,
) -> list[Record]:
    order_id = _extract_order_id(order_detail) or fallback_order_id
    if order_id is None:
        raise ValueError("Cannot extract order identifier from Rohlik order detail")

    return [
        _normalize_order_item(
            order_detail=order_detail,
            order_id=order_id,
            fallback_order_created_at=fallback_order_created_at,
            item=item,
            include_raw=include_raw,
            product_metadata_loader=product_metadata_loader,
        )
        for item in mapping_list_value(order_detail, "items")
    ]


def _normalize_order_item(
    order_detail: JsonObject,
    order_id: str | int,
    fallback_order_created_at: str | None,
    item: JsonObject,
    include_raw: bool,
    product_metadata_loader: ProductMetadataLoader | None,
) -> Record:
    product_id = int_value(item.get("id"))
    product_metadata = (
        product_metadata_loader(product_id)
        if product_id is not None and product_metadata_loader is not None
        else RohlikProductMetadata()
    )
    card = product_metadata.card or {}
    detail = product_metadata.detail or {}
    content = product_metadata.content or {}
    delivery_slot = mapping_value(order_detail, "deliverySlot")
    price_composition = mapping_value(item, "priceComposition")
    total_price = mapping_value(price_composition or {}, "total")
    unit_price = mapping_value(price_composition or {}, "unit")
    categories = _category_nodes(detail)
    main_category = categories[0] if categories else None
    leaf_category = categories[-1] if categories else None
    order_created_at = text_value(order_detail.get("orderTime")) or fallback_order_created_at
    quantity = extract_number(item.get("amount"))
    currency = money_currency(total_price) or money_currency(unit_price)

    record: Record = {
        "shop": "rohlik",
        "order_id": order_id,
        "order_number": text_value(order_detail.get("id")),
        "order_date": date_part(order_created_at),
        "order_created_at": order_created_at,
        "delivery_slot_start": text_value((delivery_slot or {}).get("since")),
        "delivery_slot_end": text_value((delivery_slot or {}).get("till")),
        "order_item_id": item.get("orderFieldId"),
        "product_id": item.get("id"),
        "product_slug": text_value(card.get("slug")),
        "product_url": _product_url(card),
        "product_name": text_value(item.get("name")) or text_value(card.get("name")),
        "brand": _brand(card, detail),
        "main_category_id": _node_id(main_category),
        "main_category": _node_name(main_category),
        "category_id": _node_id(leaf_category),
        "category": _node_name(leaf_category),
        "category_path": _category_path(categories),
        "category_ids": [_node_id(node) for node in categories if _node_id(node) is not None],
        "quantity": quantity,
        "quantity_ordered": None,
        "quantity_delivered": quantity,
        "unit": text_value(item.get("unit")),
        "package_quantity": None,
        "package_unit": text_value(item.get("textualAmount")),
        "price_total": money_amount(total_price),
        "price_unit": money_amount(unit_price),
        "price_per_unit": money_amount(unit_price),
        "price_per_unit_unit": text_value(item.get("unit")),
        "currency": currency,
        "product_description": text_value(content.get("description")),
        "ingredients_text": _ingredients_text(content),
        "product_enriched": product_metadata.card is not None
        or product_metadata.detail is not None,
    }

    if include_raw:
        record["raw"] = dict(item)

    return record


def _safe_get_json(call: Callable[[], JsonObject]) -> JsonObject | None:
    try:
        return call()
    except (ConnectionError, PermissionError, TypeError, ValueError):
        return None


def _extract_order_id(payload: JsonObject) -> str | int | None:
    value = first_value(
        payload,
        ("id", "orderNumber", "order_number", "orderId", "order_id", "uuid", "code"),
    )
    return value if isinstance(value, (str, int)) else None


def _product_url(card: JsonObject) -> str | None:
    slug = text_value(card.get("slug"))
    if slug is None:
        return None
    return f"/{slug}"


def _brand(card: JsonObject, detail: JsonObject) -> str | None:
    header = mapping_value(detail, "header")
    return (
        text_value(card.get("brand"))
        or text_value((header or {}).get("brandFilter"))
        or _brand_from_filters(detail)
    )


def _brand_from_filters(detail: JsonObject) -> str | None:
    for filter_item in materialize_mapping_list(detail.get("filters")) or []:
        slug = text_value(filter_item.get("slug"))
        filter_type = text_value(filter_item.get("type"))
        if slug not in {"znacka", "brand"} and filter_type not in {"brand", "znacka"}:
            continue

        values = materialize_mapping_list(filter_item.get("values")) or []
        if values:
            return text_value(values[0].get("name"))
    return None


def _category_nodes(product_detail: JsonObject) -> list[JsonObject]:
    nodes = _named_nodes(materialize_mapping_list(product_detail.get("breadcrumbs")))
    if nodes:
        return nodes

    product = mapping_value(product_detail, "product")
    main_category_id = (product or {}).get("mainCategoryId")
    if main_category_id is not None:
        return [{"id": main_category_id, "name": None}]

    return []


def _named_nodes(nodes: list[JsonObject] | None) -> list[JsonObject]:
    if nodes is None:
        return []
    return [node for node in nodes if _node_name(node) is not None]


def _node_id(node: JsonObject | None) -> Any:
    if node is None:
        return None
    return int_value(node.get("id")) or text_value(node.get("id")) or text_value(node.get("slug"))


def _node_name(node: JsonObject | None) -> str | None:
    if node is None:
        return None
    return text_value(node.get("name"))


def _category_path(categories: list[JsonObject]) -> list[str]:
    return compact_text_list([_node_name(node) for node in categories])


def _ingredients_text(content: JsonObject) -> str | None:
    composition = mapping_value(content, "composition")
    if composition is None:
        return None

    plain_ingredients = text_value(composition.get("plainIngredients"))
    if plain_ingredients is not None:
        return plain_ingredients

    ingredients = materialize_mapping_list(composition.get("ingredients")) or []
    titles = compact_text_list([item.get("title") for item in ingredients])
    return "; ".join(titles) if titles else None

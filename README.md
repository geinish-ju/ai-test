# AI Test Grocery Data Acquisition

Minimal project for exporting personal grocery order history into normalized product-line records.

Current scope:

- Kosik data acquisition;
- Rohlik data acquisition;
- one shared normalized order-item schema;
- product metadata enrichment for categories, category paths, descriptions, and ingredients;
- local ignored cookie files for authorization;
- sample data export for checking the workflow without real shop access;
- no model training yet.

## Install

```powershell
python -m pip install -e ".[dev]"
```

## Local Secrets

Real browser cookies are read from ignored local files:

```text
secrets/kosik.cookies.txt
secrets/rohlik.cookies.txt
```

Each file should contain one authenticated browser `Cookie` header value:

```text
cookie_a=value; cookie_b=value
```

The whole `secrets/` directory is ignored by git.

## Data Acquisition Sample

```powershell
ai-test export-sample
```

Default output:

```text
data/raw/sample_order_items.json
```

## Data Acquisition From Shops

```powershell
ai-test export-kosik
ai-test export-rohlik --include-product-content
ai-test export-all --include-product-content
```

Default outputs:

```text
data/raw/kosik_order_items.json
data/raw/rohlik_order_items.json
data/raw/grocery_order_items.json
```

Product enrichment is enabled by default. Use `--skip-product-enrichment` only when you want a
faster data acquisition run with fewer product metadata fields.

## API Flow

Kosik:

1. `https://www.kosik.cz/api/front/profile/order-list`
2. `https://www.kosik.cz/api/front/profile/order/<order_id>`
3. `https://www.kosik.cz/api/front/product/slug/<slug>`

Rohlik:

1. `https://www.rohlik.cz/api/v3/orders/delivered`
2. `https://www.rohlik.cz/api/v3/orders/<order_id>`
3. `https://www.rohlik.cz/api/v1/products/<product_id>/card`
4. `https://www.rohlik.cz/api/v1/products/<product_id>/detail`
5. `https://www.rohlik.cz/api/v1/products/<product_id>/detail/content`

## Normalized Record

Each purchased product line becomes one record. Core fields:

```json
{
  "shop": "kosik",
  "order_id": "kosik-demo-2026-06-20",
  "order_date": "2026-06-20",
  "order_created_at": "2026-06-20T10:00:00+02:00",
  "order_item_id": "kosik-line-1",
  "product_id": "sample-product-1",
  "product_slug": "sample-product-1",
  "product_url": "/sample-product-1",
  "product_name": "Example product",
  "brand": null,
  "main_category_id": "fruit-and-vegetables",
  "main_category": "Fruit and vegetables",
  "category_id": "fresh-fruit",
  "category": "Fresh fruit",
  "category_path": ["Fruit and vegetables", "Fresh fruit"],
  "quantity": 1.0,
  "quantity_ordered": 1.0,
  "quantity_delivered": 1.0,
  "unit": "pcs",
  "price_total": 49.9,
  "price_unit": 49.9,
  "price_per_unit": 49.9,
  "price_per_unit_unit": "kg",
  "currency": "CZK",
  "product_description": "Product description when the shop provides it.",
  "ingredients_text": "Ingredients when the shop provides them.",
  "product_enriched": true
}
```

The explicit Data acquisition source maps live in:

- `src/ai_testing/data_acquisition/kosik.py` as `KOSIK_FIELD_SOURCES`;
- `src/ai_testing/data_acquisition/rohlik.py` as `ROHLIK_FIELD_SOURCES`.

## Privacy

`secrets/`, `data/raw/`, and `data/processed/` are ignored by git. Do not commit cookies or exported order history.

## Quality Gate

```powershell
ruff format --check .
ruff check --no-cache .
mypy
```

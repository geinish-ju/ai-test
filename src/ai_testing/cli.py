from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

from ai_testing.data_acquisition import (
    KosikApiClient,
    KosikOrderHistoryAdapter,
    RohlikApiClient,
    RohlikOrderHistoryAdapter,
)
from ai_testing.sample_data import sample_grocery_order_item_records


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ai-test")
    subparsers = parser.add_subparsers(dest="command", required=True)

    kosik_parser = subparsers.add_parser("export-kosik", help="Export Kosik order items")
    _add_common_export_args(
        kosik_parser,
        default_output="data/raw/kosik_order_items.json",
        default_env="KOSIK_COOKIES",
        default_cookies_file="secrets/kosik.cookies.txt",
        default_base_url="https://www.kosik.cz",
    )

    rohlik_parser = subparsers.add_parser("export-rohlik", help="Export Rohlik order items")
    _add_common_export_args(
        rohlik_parser,
        default_output="data/raw/rohlik_order_items.json",
        default_env="ROHLIK_COOKIES",
        default_cookies_file="secrets/rohlik.cookies.txt",
        default_base_url="https://www.rohlik.cz",
    )
    rohlik_parser.add_argument(
        "--include-product-content",
        action="store_true",
        help="Also fetch Rohlik product detail/content for descriptions and ingredients",
    )

    all_parser = subparsers.add_parser("export-all", help="Export Kosik and Rohlik order items")
    all_parser.add_argument(
        "--output",
        default="data/raw/grocery_order_items.json",
        help="Path to write combined normalized order-item records as JSON",
    )
    all_parser.add_argument("--kosik-cookies-env", default="KOSIK_COOKIES")
    all_parser.add_argument("--rohlik-cookies-env", default="ROHLIK_COOKIES")
    all_parser.add_argument("--kosik-cookies-file", default="secrets/kosik.cookies.txt")
    all_parser.add_argument("--rohlik-cookies-file", default="secrets/rohlik.cookies.txt")
    all_parser.add_argument("--include-raw", action="store_true")
    all_parser.add_argument(
        "--skip-product-enrichment",
        action="store_true",
        help="Skip product metadata endpoints and export only order-detail data",
    )
    all_parser.add_argument(
        "--include-product-content",
        action="store_true",
        help="Also fetch Rohlik product detail/content for descriptions and ingredients",
    )

    sample_parser = subparsers.add_parser(
        "export-sample",
        help="Write sample normalized grocery order-item records",
    )
    sample_parser.add_argument(
        "--output",
        default="data/raw/sample_order_items.json",
        help="Path to write sample normalized order-item records as JSON",
    )

    args = parser.parse_args(argv)
    if args.command == "export-kosik":
        records = _fetch_kosik(args)
        _write_records(records, Path(args.output))
        return 0

    if args.command == "export-rohlik":
        records = _fetch_rohlik(args)
        _write_records(records, Path(args.output))
        return 0

    if args.command == "export-all":
        records = [
            *_fetch_kosik(
                argparse.Namespace(
                    base_url="https://www.kosik.cz",
                    cookies=None,
                    cookies_file=args.kosik_cookies_file,
                    cookies_env=args.kosik_cookies_env,
                    include_raw=args.include_raw,
                    skip_product_enrichment=args.skip_product_enrichment,
                )
            ),
            *_fetch_rohlik(
                argparse.Namespace(
                    base_url="https://www.rohlik.cz",
                    cookies=None,
                    cookies_file=args.rohlik_cookies_file,
                    cookies_env=args.rohlik_cookies_env,
                    include_raw=args.include_raw,
                    include_product_content=args.include_product_content,
                    skip_product_enrichment=args.skip_product_enrichment,
                )
            ),
        ]
        _write_records(records, Path(args.output))
        return 0

    if args.command == "export-sample":
        _write_records(sample_grocery_order_item_records(), Path(args.output))
        return 0

    return 2


def _add_common_export_args(
    parser: argparse.ArgumentParser,
    default_output: str,
    default_env: str,
    default_cookies_file: str,
    default_base_url: str,
) -> None:
    parser.add_argument(
        "--output",
        default=default_output,
        help="Path to write normalized order-item records as JSON",
    )
    parser.add_argument(
        "--cookies",
        help="Authenticated browser Cookie header value. Prefer --cookies-file for local use.",
    )
    parser.add_argument(
        "--cookies-file",
        default=default_cookies_file,
        help="Local ignored file containing the authenticated browser Cookie header",
    )
    parser.add_argument(
        "--cookies-env",
        default=default_env,
        help="Environment variable containing the authenticated browser Cookie header",
    )
    parser.add_argument("--base-url", default=default_base_url)
    parser.add_argument("--include-raw", action="store_true")
    parser.add_argument(
        "--skip-product-enrichment",
        action="store_true",
        help="Skip product metadata endpoints and export only order-detail data",
    )


def _fetch_kosik(args: argparse.Namespace) -> list[dict[str, object]]:
    cookies = _resolve_cookies(args.cookies, args.cookies_file, args.cookies_env)
    client = KosikApiClient(base_url=args.base_url, cookies=cookies)
    return KosikOrderHistoryAdapter(
        client,
        include_raw=bool(args.include_raw),
        enrich_products=not bool(args.skip_product_enrichment),
    ).fetch_order_item_records()


def _fetch_rohlik(args: argparse.Namespace) -> list[dict[str, object]]:
    cookies = _resolve_cookies(args.cookies, args.cookies_file, args.cookies_env)
    client = RohlikApiClient(base_url=args.base_url, cookies=cookies)
    return RohlikOrderHistoryAdapter(
        client,
        include_raw=bool(args.include_raw),
        enrich_products=not bool(args.skip_product_enrichment),
        include_product_content=bool(getattr(args, "include_product_content", False)),
    ).fetch_order_item_records()


def _write_records(records: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output_path),
                "record_count": len(records),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _resolve_cookies(cookies: str | None, cookies_file: str | None, env_name: str) -> str:
    if cookies:
        return cookies

    if cookies_file:
        file_value = _read_cookies_file(Path(cookies_file))
        if file_value:
            return file_value

    env_value = os.getenv(env_name)
    if env_value:
        return env_value

    raise SystemExit(
        "Missing cookies. Put an authenticated Cookie header into "
        f"{cookies_file}, set {env_name}, or pass --cookies."
    )


def _read_cookies_file(path: Path) -> str | None:
    if not path.exists():
        return None

    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return " ".join(lines) if lines else None


if __name__ == "__main__":
    raise SystemExit(main())

"""
Data loading and indexing module for Buy or Wait? AI financial agent.

Pre-indexes CSV datasets at startup for O(1) per-request node lookups,
handles exchange rate conversions, and provides cached access to image/message content.
"""

from __future__ import annotations
import csv
import math
from datetime import datetime
from typing import Any
import config


class DataLoader:
    def __init__(self, dataset_dir: str | None = None):
        self.dataset_dir = dataset_dir or str(config.DATASET_DIR)
        
        self.requests: list[dict[str, Any]] = []
        self.profiles: dict[str, dict[str, Any]] = {}
        self.events_by_user: dict[str, list[dict[str, Any]]] = {}
        self.messages_by_user: dict[str, list[dict[str, Any]]] = {}
        self.messages_by_request: dict[str, list[dict[str, Any]]] = {}
        self.images_by_user: dict[str, list[dict[str, Any]]] = {}
        self.images_by_request: dict[str, list[dict[str, Any]]] = {}
        self.payment_options_by_request: dict[str, list[dict[str, Any]]] = {}
        self.exchange_rates: list[dict[str, Any]] = []
        
        self.load_all()

    def _read_csv(self, file_path: str) -> list[dict[str, Any]]:
        with open(file_path, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            return [dict(row) for row in reader]

    def load_all(self) -> None:
        # 1. Requests
        self.requests = self._read_csv(str(config.REQUESTS_CSV))
        for r in self.requests:
            r["requested_amount"] = float(r["requested_amount"]) if r.get("requested_amount") else 0.0
            r["allows_partial_payment"] = str(r.get("allows_partial_payment", "")).lower() == "true"

        # 2. Profiles
        profiles_raw = self._read_csv(str(config.PROFILES_CSV))
        for p in profiles_raw:
            user_id = p["user_id"]
            p["current_available_balance"] = float(p["current_available_balance"]) if p.get("current_available_balance") else 0.0
            p["minimum_balance_to_keep"] = float(p["minimum_balance_to_keep"]) if p.get("minimum_balance_to_keep") else 0.0
            max_inst = p.get("max_installment_months")
            p["max_installment_months"] = float(max_inst) if max_inst and max_inst.strip() != "" else None
            
            # Helper list parser for semicolon or comma separated preferences
            p["categories_to_protect"] = [c.strip() for c in p.get("expense_categories_to_protect", "").split("|") if c.strip()]
            p["categories_to_reduce"] = [c.strip() for c in p.get("expense_categories_user_is_willing_to_reduce", "").split("|") if c.strip()]
            p["categories_to_stop"] = [c.strip() for c in p.get("expense_categories_user_is_willing_to_stop", "").split("|") if c.strip()]
            p["payment_methods_to_consider"] = [m.strip() for m in p.get("payment_methods_user_will_consider", "").split("|") if m.strip()]
            
            self.profiles[user_id] = p

        # 3. Events
        events_raw = self._read_csv(str(config.EVENTS_CSV))
        for ev in events_raw:
            u_id = ev["user_id"]
            amt_str = ev.get("amount")
            ev["amount"] = float(amt_str) if amt_str and amt_str.strip() not in ("", "null", "NaN") else None
            min_amt_str = ev.get("minimum_allowed_amount")
            ev["minimum_allowed_amount"] = float(min_amt_str) if min_amt_str and min_amt_str.strip() not in ("", "null", "NaN") else None
            
            if u_id not in self.events_by_user:
                self.events_by_user[u_id] = []
            self.events_by_user[u_id].append(ev)

        # 4. Exchange Rates
        rates_raw = self._read_csv(str(config.EXCHANGE_RATES_CSV))
        for r in rates_raw:
            r["rate"] = float(r["rate"])
            self.exchange_rates.append(r)

        # 5. Payment Options
        opts_raw = self._read_csv(str(config.PAYMENT_OPTIONS_CSV))
        for opt in opts_raw:
            req_id = opt["request_id"]
            opt["payment_amount"] = float(opt["payment_amount"]) if opt.get("payment_amount") else 0.0
            opt["number_of_payments"] = int(opt["number_of_payments"]) if opt.get("number_of_payments") else 1
            opt["payment_frequency_days"] = int(opt["payment_frequency_days"]) if opt.get("payment_frequency_days") else 0
            opt["financing_fee"] = float(opt["financing_fee"]) if opt.get("financing_fee") else 0.0
            opt["total_payable_amount"] = float(opt["total_payable_amount"]) if opt.get("total_payable_amount") else opt["payment_amount"] * opt["number_of_payments"]
            
            if req_id not in self.payment_options_by_request:
                self.payment_options_by_request[req_id] = []
            self.payment_options_by_request[req_id].append(opt)

        # 6. Messages
        msgs_raw = self._read_csv(str(config.MESSAGES_CSV))
        for m in msgs_raw:
            u_id = m.get("user_id")
            r_id = m.get("request_id")
            if u_id:
                if u_id not in self.messages_by_user:
                    self.messages_by_user[u_id] = []
                self.messages_by_user[u_id].append(m)
            if r_id:
                if r_id not in self.messages_by_request:
                    self.messages_by_request[r_id] = []
                self.messages_by_request[r_id].append(m)

        # 7. Images
        imgs_raw = self._read_csv(str(config.IMAGES_CSV))
        for img in imgs_raw:
            u_id = img.get("user_id")
            r_id = img.get("request_id")
            if u_id:
                if u_id not in self.images_by_user:
                    self.images_by_user[u_id] = []
                self.images_by_user[u_id].append(img)
            if r_id:
                if r_id not in self.images_by_request:
                    self.images_by_request[r_id] = []
                self.images_by_request[r_id].append(img)

    def convert_currency(self, amount: float, from_curr: str, to_curr: str, date_str: str) -> tuple[float, str | None]:
        """
        Converts amount from from_curr to to_curr using exchange_rates.csv.
        Returns (converted_amount, warning_msg_or_none).
        """
        if from_curr == to_curr or amount == 0.0:
            return amount, None

        # Filter available exchange rates for pair or inverse pair
        matches = [
            r for r in self.exchange_rates
            if (r["from_currency"] == from_curr and r["to_currency"] == to_curr) or
               (r["from_currency"] == to_curr and r["to_currency"] == from_curr)
        ]

        if not matches:
            # Missing conversion rate fallback
            return amount, f"No exchange rate found for {from_curr} -> {to_curr}. Used 1:1."

        # Find match on or closest prior to date_str
        exact_or_prior = [m for m in matches if m["rate_date"] <= date_str]
        selected_rate_row = max(exact_or_prior, key=lambda x: x["rate_date"]) if exact_or_prior else min(matches, key=lambda x: x["rate_date"])

        rate = selected_rate_row["rate"]
        warning = None
        if not exact_or_prior:
            warning = f"Exchange rate for {from_curr}->{to_curr} on {date_str} fallback to date {selected_rate_row['rate_date']}"

        if selected_rate_row["from_currency"] == from_curr:
            converted = amount * rate
        else:
            converted = amount / rate if rate != 0 else amount

        return converted, warning


# Global instance
_default_data_loader: DataLoader | None = None

def get_data_loader() -> DataLoader:
    global _default_data_loader
    if _default_data_loader is None:
        _default_data_loader = DataLoader()
    return _default_data_loader

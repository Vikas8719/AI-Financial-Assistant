"""SEC EDGAR service — search and fetch company filings."""
import httpx
from typing import Optional


EDGAR_BASE = "https://efts.sec.gov/LATEST/search-index"
EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index?q={query}&dateRange=custom&startdt={start}&enddt={end}&forms={form}"
EDGAR_API = "https://data.sec.gov"


class SecEdgarService:

    async def search_filings(self, company_name: str, filing_type: str = "10-K") -> dict:
        """Search SEC EDGAR for company filings."""
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                # Search for company CIK
                search_url = f"https://efts.sec.gov/LATEST/search-index?q=%22{company_name}%22&forms={filing_type}"
                response = await client.get(
                    f"https://efts.sec.gov/LATEST/search-index",
                    params={
                        "q": f'"{company_name}"',
                        "forms": filing_type,
                        "dateRange": "custom",
                        "startdt": "2023-01-01"
                    },
                    headers={"User-Agent": "FinBot research@finbot.ai"}
                )

                if response.status_code != 200:
                    # Fallback to full-text search
                    return await self._fulltext_search(company_name, filing_type)

                data = response.json()
                hits = data.get("hits", {}).get("hits", [])[:5]

                filings = []
                for hit in hits:
                    src = hit.get("_source", {})
                    filings.append({
                        "company": src.get("entity_name", company_name),
                        "filing_type": src.get("file_type", filing_type),
                        "filed_date": src.get("file_date"),
                        "period": src.get("period_of_report"),
                        "description": src.get("form_type"),
                        "url": f"https://www.sec.gov/Archives/edgar/data/{src.get('entity_id', '')}/{src.get('file_num', '')}"
                    })

                return {
                    "company": company_name,
                    "filing_type": filing_type,
                    "total_found": data.get("hits", {}).get("total", {}).get("value", 0),
                    "filings": filings
                }
        except Exception as e:
            return {"error": str(e), "company": company_name}

    async def _fulltext_search(self, company_name: str, filing_type: str) -> dict:
        """Fallback: search EDGAR full-text search."""
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(
                    "https://efts.sec.gov/LATEST/search-index",
                    params={"q": company_name, "forms": filing_type},
                    headers={"User-Agent": "FinBot research@finbot.ai"}
                )
                data = response.json()
                hits = data.get("hits", {}).get("hits", [])[:3]
                filings = []
                for hit in hits:
                    src = hit.get("_source", {})
                    filings.append({
                        "company": src.get("entity_name"),
                        "filing_type": src.get("file_type"),
                        "filed_date": src.get("file_date"),
                        "url": f"https://www.sec.gov{src.get('file_num', '')}"
                    })
                return {"filings": filings, "company": company_name}
        except Exception as e:
            return {"error": str(e)}

    async def get_company_facts(self, cik: str) -> dict:
        """Get structured company financial facts from EDGAR API."""
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(
                    f"{EDGAR_API}/api/xbrl/companyfacts/CIK{cik.zfill(10)}.json",
                    headers={"User-Agent": "FinBot research@finbot.ai"}
                )
                if response.status_code == 200:
                    data = response.json()
                    facts = data.get("facts", {}).get("us-gaap", {})
                    # Extract key metrics
                    result = {}
                    for metric in ["Revenues", "NetIncomeLoss", "Assets", "EarningsPerShareBasic"]:
                        if metric in facts:
                            units = facts[metric].get("units", {})
                            for unit_key, values in units.items():
                                if values:
                                    latest = sorted(values, key=lambda x: x.get("end", ""))[-1]
                                    result[metric] = {"value": latest.get("val"), "unit": unit_key, "period": latest.get("end")}
                    return result
                return {"error": "Not found"}
        except Exception as e:
            return {"error": str(e)}

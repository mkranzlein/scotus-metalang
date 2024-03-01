import sqlite3

import aiohttp


# Get docket numbers to cross-reference
connection = sqlite3.connect("api_log.db")
# Insert all docket numbers into cases
with connection:
    rows = connection.execute("""--sql
                              SELECT docket_number from cases
                              """).fetchall()

docket_numbers = [row[0] for row in rows]


# Get metadata for each docket number
async def main():
    connector = aiohttp.TCPConnector(limit_per_host=10)
    async with aiohttp.ClientSession(connector=connector) as session:
        for docket_number in docket_numbers:
            # TODO: figure out what term case was part of based on decision
            request_url = f"'https://api.oyez.org/cases/{year}/{docket_number}'"
            async with session.get(request_url) as response:
                return await response.json()

# Save metadata for each docket number
# TODO

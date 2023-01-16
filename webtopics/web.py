import asyncio
import pickle
import time

import aiohttp
import pandas as pd

from collections import namedtuple

Website = namedtuple('Website', ['url', 'title', 'content'])


async def fetch(session, url):
    try:
        async with session.get(url) as response:
            return await response.text()
    except Exception as e:
        return None


async def fetch_all(urls):
    connector = aiohttp.TCPConnector(limit_per_host=5, limit=500)
    async with aiohttp.ClientSession(connector=connector) as session:
        results = await asyncio.gather(
            *[fetch(session, url) for url in urls],
            return_exceptions=False
        )
        return results


if __name__ == '__main__':
    data = pd.read_parquet('data/marketing_sample_walmart.parq.gzip')
    urls = data['Product URL']

    start = time.time()
    res = asyncio.run(fetch_all(urls))
    duration = time.time() - start

    print(f"Downloaded {len(urls)} sites in {round(duration/60, 1)} minutes.")
    start = time.time()
    data['url_text'] = [get_text_content(x) for x in res]
    duration = time.time() - start
    print(f"Extracted text from {len(urls)} sites in {round(duration / 60, 1)} minutes.")
    ouput_fn = 'bookmarks_data.p'
    dump(data, ouput_fn)

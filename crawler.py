from playwright.async_api import async_playwright
import requests
import time 
import random
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import random
# 使用异步抓取
import asyncio
import csv
from urllib.parse import urljoin, urlparse
import aiofiles
from datetime import datetime
import akshare as ak

# 深度爬虫示例代码


class AsyncPlaywrightCrawler:
    def __init__(self, start_url, max_concurrency=15, max_pages=10000):
        self.start_url = start_url
        self.max_concurrency = max_concurrency
        self.max_pages = max_pages
        self.visited = set()
        self.queue = asyncio.Queue()
        self.queue.put_nowait(start_url)
        self.data = []
        self.domain = urlparse(start_url).netloc
        self.gnextpage = 100  # Initialize Gnextpage as an instance variable
        self.lock = asyncio.Lock()  # Create a lock for safe updates


    def is_valid_url(self, url):
        parsed = urlparse(url)
        return parsed.netloc == self.domain and parsed.scheme in ('http', 'https')
    

    async def fetch_page(self, page):
        try:
            await page.goto(self.current_url, timeout=30_000)
            return page
        except Exception as e:
            print(f"Error loading {self.current_url}: {e}")
            return None

    async def extract_data(self, page):
        data = []
        # 获取 ul 元素的文本内容
        table = page.locator("table.default_list")
        if not table:
            print("Table not found")
            return None
        

        # 等待表格加载完成
        await table.wait_for(state="visible", timeout=10_000)
        
        # 获取表头和表体
        headers = await table.locator("thead.listhead").all()
        rows = await table.locator("tbody.listbody").all()
         # 构建表头映射
        header_map = [await header.inner_text() for header in headers]
        # clearprint(header_map)
        header_map = header_map[0].split("\n\t\n")
        # print(header_map)
        # 遍历并打印数据
        
        for row in rows:
            cells = await row.locator("tr.listitem").all()
            for  cell in cells:
                # 获取每个单元格的文本内容
                cell_text = await cell.inner_text()
                #rint(cell_text)
                 # Create a dictionary with appropriate keys
                cell_texts = cell_text.split("\n\t\n")


                raw_time = cell_texts[-1]  # 假设时间在最后一列
                try:
                    # 补充年份信息并解析时间
                    current_year = datetime.now().year
                    parsed_time = datetime.strptime(f"{current_year}-{raw_time}", "%Y-%m-%d %H:%M")
                    formatted_time = parsed_time.strftime("%Y-%m-%d %H:%M:%S")  # 转换为标准格式
                except ValueError as e:
                    print(f"Failed to parse time: {raw_time}, error: {e}")
                    formatted_time = None

                row_data = {
                    "created_time": formatted_time,  # Replace "column_1" with a meaningful key
                    "title": cell_texts[2]   # Replace "column_2" with a meaningful key
                }
                # print(row_data)
                data.append(row_data)
                # 将每一行数据存储为字典'
        return data

       
        


    async def extract_links(self, page):
        # 提取当前页面所有有效链接
        links = set()
        for a_tag in await page.query_selector_all("a"):
            href = await a_tag.get_attribute("href")
            full_url = urljoin(self.start_url, href)
            if self.is_valid_url(full_url):
                links.add(full_url)
        return links

    async def worker(self):
        # 创建异步任务池
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=False,  # 设置为 False 以查看浏览器操作
                slow_mo=50,  # 设置慢速操作以便观察 
                args=["--no-sandbox", "--disable-dev-shm-usage", "--autoplay-policy=no-user-gesture-required",
                "--disable-notifications", "--disable-popup-blocking", "--disable-infobars"]
            )
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) ..."
            )

           
            try:
                while True:
                    # 从队列中获取 URL
                    if self.queue.empty():
                        break
                    self.current_url = await self.queue.get()
                    if self.current_url in self.visited or len(self.visited) >= self.max_pages:
                        self.queue.task_done()
                        continue

                    print(f"[*] Crawling: {self.current_url}")
                    self.visited.add(self.current_url)

                    page = await context.new_page()
                    try:
                        # 设置页面大小  
                        await page.set_viewport_size({"width": 1280, "height": 800})
                        # 设置请求头
                        await page.set_extra_http_headers({
                                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ...",
                                "Accept-Language": "zh-CN,zh;q=0.9",
                                "Accept-Encoding": "gzip, deflate, br",
                                "Connection": "keep-alive"
                        })
                        print('page',page)
                        await self.queue.put(self.current_url)  # 重新放回队列以处理动态更新
                        # 等待页面加载完成
                        await self.fetch_page(page)
                        # 等待一段时间以模拟人类行为
                        await asyncio.sleep(random.uniform(1, 3))

                        # 处理页面数据
                        data = await self.extract_data(page)
                        if data:
                            print(f"[*] Extracted data from {self.current_url}")
                            # 处理数据（例如保存到文件或数据库）
                            # 这里我们简单地打印出来
                            # print(data)
                            # 将数据添加到列表中
                            self.data.append(data)

                        '''
                        # 提取链接并加入队列所有遍历暂时不用保留
                        links = await self.extract_links(page)
                        for link in links:
                            if link not in self.visited:
                                await self.queue.put(link)
                        '''
                        # Safely update Gnextpage and generate the next link
                        async with self.lock:
                            if len(self.visited) < self.max_pages:  # 确保不超过最大页数
                                self.gnextpage += 10
                                next_link = f"https://guba.eastmoney.com/list,zssh000001_{self.gnextpage}.html"
                                print(f"[*] Next page link: {next_link}")
                                if next_link not in self.visited:
                                    await self.queue.put(next_link)
                                    print(f"[*] The next website is {next_link}")
                    finally:
                        await page.close() 

                    self.queue.task_done()

            except Exception as e:
                print(f"Failed to process {self.current_url}: {e}")
                self.queue.task_done()
            finally:  
                await context.close()    
                await browser.close()
  

    async def run(self):
        # 创建多个工作线程
        workers = [
            asyncio.create_task(self.worker())
            for _ in range(self.max_concurrency)
        ]
        try:
            await self.queue.join()  # 等待队列处理完成
        finally:
            for w in workers:
                w.cancel()  # 取消所有任务
            await asyncio.gather(*workers, return_exceptions=True)  # 等待所有任务完成

        
        '''
        print(f"[*] Crawling finished. Total pages visited: {len(self.visited)}")
        print(f"[*] Data collected: {len(self.data)} items")
        # 打印数据
        for i, item in enumerate(self.data):
            print(f"[*] Data {i + 1}: {item}")
        '''

        # 保存数据到 CSV
        async with aiofiles.open("./data/stock_comments_seg.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["created_time", "title"])
            await writer.writeheader()
             # 打印数据
            for i, item in enumerate(self.data):
                #print(f"[*] Data {i + 1}: {item}")
                for row in item:
                    if isinstance(row, dict):
                        await writer.writerow(row)

    def save_to_csv(self, filename):
        df = pd.DataFrame(self.data)
        df.to_csv(filename, index=False)

def getSH1Daily():
        # 获取上证综指（sh00001）的日线数据
    df = ak.stock_zh_a_hist(symbol="000001", period="daily", start_date="20160101", end_date="20250401")
    df.drop(['股票代码','成交额','振幅','涨跌幅','涨跌额','换手率'], axis=1, inplace=True)
    new_columns = ['date','open','high','low','close','volume']
    df.columns = new_columns
    print(df)
    df.to_csv("./data/sh000001.csv", index=False)

#Run the crawler
if __name__ == "__main__":


    #getSH1Daily()


    # 1. 设置起始 URL
    start_url = "https://guba.eastmoney.com/list,zssh000001.html"
    
    # 2. 创建异步爬虫实例
    crawler = AsyncPlaywrightCrawler(start_url, max_concurrency=15)
    
    # 3. 启动爬虫
    asyncio.run(crawler.run())




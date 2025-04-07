from playwright.async_api import async_playwright
import os
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
import jieba
import re
# 深度爬虫示例代码


class AsyncPlaywrightCrawler:
    def __init__(self, start_url, max_concurrency=15, max_pages=10000):
        self.gnextpage = 3800  # Initialize Gnextpage as an instance variable
        start_url = f"https://guba.eastmoney.com/list,zssh000001_{self.gnextpage}.html"
        self.start_url = start_url
        self.max_concurrency = max_concurrency
        self.max_pages = max_pages
        self.visited = set()
        self.queue = asyncio.Queue()
        self.queue.put_nowait(start_url)
        self.data = []
        self.domain = urlparse(start_url).netloc
        self.intelvar = 5
        self.lock = asyncio.Lock()  # Create a lock for safe updates
        self.buffer = []  # 添加缓冲区
        self.buffer_size = 100  # 设置缓冲区大小


    def is_valid_url(self, url):
        parsed = urlparse(url)
        return parsed.netloc == self.domain and parsed.scheme in ('http', 'https')
    

    async def fetch_page(self, page):
        try:
            await page.goto(self.current_url, timeout=10_000)
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
        
        #清理数据
        for row in rows:
            cells = await row.locator("tr.listitem").all()
            for  cell in cells:
                try:
                    # 获取每个单元格的文本内容
                    cell_text = await cell.inner_text()
                    #rint(cell_text)
                    # Create a dictionary with appropriate keys
                    cell_texts = cell_text.split("\n\t\n")
                    raw_time = cell_texts[-1]  # 假设时间在最后一列
                    # 补充年份信息并解析时间
                    current_year = datetime.now().year
                    parsed_time = datetime.strptime(f"{current_year}-{raw_time}", "%Y-%m-%d %H:%M")
                    formatted_time = parsed_time.strftime("%Y-%m-%d %H:%M:%S")  # 转换为标准格式
                    # 执行分词
                    cell_texts[2] = re.sub(r'[^\u4e00-\u9fff\w\s]', '', cell_texts[2])
                    words = jieba.cut(cell_texts[2])
                    # 过滤空白词并拼接
                    extractwolds = ' '.join([word.strip() for word in words if word.strip()])
                    row_data = {
                        "created_time": formatted_time,  # Replace "column_1" with a meaningful key
                        "title": extractwolds   # Replace "column_2" with a meaningful key
                    }
                    # print(row_data)
                    data.append(row_data)
                    # 将每一行数据存储为字典'
                except ValueError as e:
                    print(f"Failed to clear data , error: {e} ")
                    continue
        return data

    async def extract_wolds(text):
        # 使用正则表达式提取中文字符和常用标点符号
        # 执行分词
        words = jieba.cut(text)
        # 过滤空白词并拼接
        return ' '.join([word.strip() for word in words if word.strip()])


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
                headless=True,  # 设置为 False 以查看浏览器操作
                #slow_mo=50,  # 设置慢速操作以便观察 
                args=["--no-sandbox", "--disable-dev-shm-usage", "--autoplay-policy=no-user-gesture-required",
                "--disable-notifications", "--disable-popup-blocking", "--disable-infobars"]
            )
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) ..."
            )
            # 禁用图片和样式表加载
            await context.route("**/*", lambda route: route.abort() if route.request.resource_type in ["image", "stylesheet"] else route.continue_())
           
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
                                "User-Agent": "Chrome/135.0.7049.42",
                                "Accept-Language": "zh-CN,zh;q=0.9",
                                "Accept-Encoding": "gzip, deflate, br",
                                "Connection": "keep-alive"
                        })
                        print('page',page)
                        await self.queue.put(self.current_url)  # 重新放回队列以处理动态更新
                        # 等待页面加载完成
                        await self.fetch_page(page)
                        # 等待一段时间以模拟人类行为
                        await asyncio.sleep(random.uniform(1, 13))
                        data=[]
                        # 处理页面数据
                        data = await self.extract_data(page)
                        if data:
                            print(f"[*] Extracted data from {self.current_url}")
                            self.buffer.extend(data)
                            if len(self.buffer) >= self.buffer_size:
                                await self.save_to_csv()
                                #await self.buffer.clear()


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
                                self.intelvar = random.randint(1, 10)
                                self.gnextpage -= self.intelvar
                                next_link= f"https://guba.eastmoney.com/list,zssh000001_{self.gnextpage}.html"
                                if self.gnextpage < 0:
                                    self.gnextpage = 0
                                    next_link = f"https://guba.eastmoney.com/list,zssh000001.html"
                                print(f"[*] Next page link: {next_link}")
                                if next_link not in self.visited:
                                    await self.queue.put(next_link)
                                    print(f"[*] The next website is {next_link}")
                    except Exception as e:
                        print(f"While trueFailed to process {self.current_url}: {e}")
                        await page.close()
                        self.queue.task_done()
                        continue
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
        

        # 保存数据到 CSV
        async with aiofiles.open("./data/stock_comments_seg.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["created_time", "title"])
            await writer.writeheader()
             # 打印数据
            for i, item in enumerate(self.data):
                #print(f"[*] Data {i + 1}: {item}")
                for row in item:
                    if isinstance(row, dict):
                        await writer.writerow(row)'
        '''

    def save_to_csv11(self, filename):
        df = pd.DataFrame(self.data)
        df.to_csv(filename, index=False)

    async def save_to_csv(self, filename="./data/stock_comments_seg.csv"):
        try:
            # 检查文件是否存在
            file_exists = os.path.exists(filename)


            # 使用异步上下文管理器打开文件
            async with aiofiles.open(
                filename,
                mode="a",  # 改为追加模式，保留历史数据
                newline="",
                encoding="utf-8"
            ) as f:
                if not file_exists:
                    print(f"文件bucunzai:")
                    writer = csv.DictWriter(f, fieldnames=["created_time", "title"])
                    await f.write(",".join(writer.fieldnames) + "\n")
                 # 写入缓冲区数据
                writer = csv.DictWriter(f, fieldnames=["created_time", "title"])
                if self.buffer:
                    for row in self.buffer:
                        await f.write(",".join([str(row[field]) for field in writer.fieldnames]) + "\n")
                    self.buffer.clear()  # 清空缓冲区
        except Exception as e:
            print(f"文件保存失败: {str(e)}")
            self.buffer.clear()  # 清空缓冲区
            # 可在此处添加重试逻辑或错误上报

def getSH1Daily():
        # 获取上证综指（sh00001）的日线数据
    df = ak.stock_zh_a_hist(symbol="000001", period="daily", start_date="20250101", end_date="20250401")
    df.drop(['股票代码','成交额','振幅','涨跌幅','涨跌额','换手率'], axis=1, inplace=True)
    new_columns = ['date','open','high','low','close','volume']
    df.columns = new_columns
    print(df)
    df.to_csv("./data/sh000001.csv", index=False)

#Run the crawler
if __name__ == "__main__":


    # 0. 获取上证综指（sh00001）的日线数据 可选
    #getSH1Daily()


    # 1. 设置起始 URL
    start_url = "https://guba.eastmoney.com/list,zssh000001.html"
    
    # 2. 创建异步爬虫实例
    crawler = AsyncPlaywrightCrawler(start_url, max_concurrency=15)
    
    # 3. 启动爬虫
    asyncio.run(crawler.run())




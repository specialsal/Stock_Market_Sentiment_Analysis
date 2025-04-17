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
import requests
# 深度爬虫示例代码


class AsyncPlaywrightCrawler:
    def __init__(self, start_url, max_concurrency=15, max_pages=10000):
        self.gnextpage = 3980  # Initialize Gnextpage as an instance variable
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
    

    async def fetch_page(self, page,headers):
        try:
            try:
                session = requests.Session()
                session.headers.update(headers)
                resp = session.get(self.current_url, timeout=500)
                resp.raise_for_status()  # 检查请求是否成功
                #print(resp.text)
                if resp.status_code != 200:
                    print(f"Failed to fetch {self.current_url}: {resp.status_code}")
                    input("Press Enter to continue...")
                soup = BeautifulSoup(resp.text, 'html.parser')
        
                # 查找 default_list 表格
                table = soup.find('table', class_='default_list')
                if not table:
                    print("Table not found in response")
                    return None
                    
                data = []
                try:
                    # 获取表头
                    headers = []
                    thead = table.find('thead', class_='listhead')
                    if thead:
                        headers = [th.get_text(strip=True) for th in thead.find_all('th')]

                    # 获取表格数据
                    tbody = table.find('tbody', class_='listbody')
                    if tbody:
                        rows = tbody.find_all('tr', class_='listitem')
                        for row in rows:
                            try:
                                # 获取行数据
                                cells = [cell.get_text(strip=True) for cell in row.find_all('td')]
                                
                                if cells:
                                    # 处理时间（假设在最后一列）
                                    raw_time = cells[-1]
                                    try:
                                        current_year = datetime.now().year
                                        parsed_time = datetime.strptime(f"{current_year}-{raw_time}", "%Y-%m-%d %H:%M")
                                        formatted_time = parsed_time.strftime("%Y-%m-%d %H:%M:%S")
                                    except ValueError as e:
                                        print(f"Error parsing time {raw_time}: {e}")
                                        formatted_time = raw_time

                                    # 处理标题（假设在第3列）
                                    title_text = cells[2] if len(cells) > 2 else ""
                                    cleaned_title = re.sub(r'[^\u4e00-\u9fff\w\s]', '', title_text)
                                    words = jieba.cut(cleaned_title)
                                    extracted_words = ' '.join([word.strip() for word in words if word.strip()])

                                    # 构建数据字典
                                    row_data = {
                                        "created_time": formatted_time,
                                        "title": extracted_words,
                                        "raw_title": title_text,
                                        "full_data": cells
                                    }
                                    
                                    if headers:
                                        # 如果有表头，添加对应的键值对
                                        row_data.update(dict(zip(headers, cells)))
                                    
                                    data.append(row_data)
                                    
                            except Exception as e:
                                print(f"Error processing row: {e}")
                                continue

                        print(f"Successfully processed {len(data)} rows")
                    else:
                        print("No table body found")

                except Exception as e:
                    print(f"Error processing table: {e}")
                    return None

                return data

            except requests.exceptions.RequestException as e:
                print(f"Request error: {e}")
                return None
            except Exception as e:
                print(f"Unexpected error: {e}")
                return None
            except requests.exceptions.RequestException as e:
                print(f"请求错误: {e}")
        except Exception as e:
            print(f"Error loading {self.current_url}: {e}")
            return None

    async def extract_data(self, page):
        try:
            data = []
            # 1. 等待表格加载并获取表格元素
            try:
                table = await page.wait_for_selector("table.default_list", timeout=5000)
                if not table:
                    print("Table not found")
                    return None
            except Exception as e:
                print(f"Error finding table: {e}")
                return None

            # 2. 获取表头
            try:
                thead = await table.query_selector("thead.listhead")
                if thead:
                    header_text = await thead.inner_text()
                    headers = [h.strip() for h in header_text.split("\n\t\n") if h.strip()]
                    print(f"Headers found: {headers}")
                else:
                    print("No headers found")
                    headers = []
            except Exception as e:
                print(f"Error processing headers: {e}")
                headers = []

            # 3. 获取所有数据行
            try:
                rows = await table.query_selector_all("tbody.listbody tr.listitem")
                if not rows:
                    print("No data rows found")
                    return None

                # 4. 处理每一行数据
                for row in rows:
                    try:
                        # 获取行文本
                        row_text = await row.inner_text()
                        cells = [cell.strip() for cell in row_text.split("\n\t\n") if cell.strip()]
                        
                        # 处理时间
                        raw_time = cells[-1]  # 假设时间在最后一列
                        try:
                            current_year = datetime.now().year
                            parsed_time = datetime.strptime(f"{current_year}-{raw_time}", "%Y-%m-%d %H:%M")
                            formatted_time = parsed_time.strftime("%Y-%m-%d %H:%M:%S")
                        except ValueError as e:
                            print(f"Error parsing time {raw_time}: {e}")
                            formatted_time = raw_time

                        # 处理标题文本（第3列，索引2）
                        if len(cells) > 2:
                            title_text = cells[2]
                            # 清理标题文本
                            cleaned_title = re.sub(r'[^\u4e00-\u9fff\w\s]', '', title_text)
                            # 分词处理
                            words = jieba.cut(cleaned_title)
                            extracted_words = ' '.join([word.strip() for word in words if word.strip()])
                        else:
                            extracted_words = ""

                        # 构建数据字典
                        row_data = {
                            "created_time": formatted_time,
                            "title": extracted_words
                            #"raw_title": title_text if len(cells) > 2 else "",  # 保存原始标题
                            #"full_data": cells  # 保存完整数据
                        }
                        
                        data.append(row_data)
                        
                    except Exception as e:
                        print(f"Error processing row: {e}")
                        continue

                print(f"Successfully processed {len(data)} rows")
                return data

            except Exception as e:
                print(f"Error processing table body: {e}")
                return None

        except Exception as e:
            print(f"Unexpected error in extract_data: {e}")
            return None

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
                user_agent='User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36'
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
                    headers = {
                                'User-Agent': 'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36',
                                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
                                'Accept-Encoding': 'gzip, deflate, br, zstd',
                                'Accept-Language': 'zh-CN,zh;q=0.9',
                                'Cache-Control': 'max-age=0',
                                'Connection': 'keep-alive',
                                'Cookie': 'qgqp_b_id=f67ccf209c2668010ce0820c6c9f2270; HAList=ty-1-000001-%u4E0A%u8BC1%u6307%u6570%2Cty-0-000722-%u6E56%u5357%u53D1%u5C55; fullscreengg=1; fullscreengg2=1; st_si=09540399922141; st_asi=delete; st_pvi=08238370541624; st_sp=2025-04-03%2013%3A52%3A05; st_inirUrl=https%3A%2F%2Fguba.eastmoney.com%2Fnews%2Cof008515%2C1535755240.html; st_sn=4; st_psi=2025041714474175-117001356556-7311554884',
                                'Host': 'guba.eastmoney.com',
                                'Sec-Fetch-Dest': 'document',
                                'Sec-Fetch-Mode': 'navigate',
                                'Sec-Fetch-Site': 'none',
                                'Sec-Fetch-User': '?1'
                    }

                    try:
                        # 设置页面大小  
                        await page.set_viewport_size({"width": 1280, "height": 800})
                        # 设置请求头
                        await page.set_extra_http_headers(headers)
                       
                        await self.queue.put(self.current_url)  # 重新放回队列以处理动态更新
                        # 等待页面加载完成
                        data=[]
                        data =  await self.fetch_page(page,headers)
                        # 等待一段时间以模拟人类行为
                        await asyncio.sleep(random.uniform(1, 13))

                        if data:
                            print(f"[*] Extracted data from {self.current_url}")
                            self.buffer.extend(data)
                            if len(self.buffer) >= self.buffer_size:
                                await self.save_to_csv()
                                #await self.buffer.clear()
                        print('save_to_csv')

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
                        print(f"While true Failed to process {self.current_url}: {e}")
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




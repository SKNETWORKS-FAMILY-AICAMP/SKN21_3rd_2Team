from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
import time

def crawl_and_save(channel_url, target_count=40, file_name="url_list.txt"):
    # 브라우저 옵션 설정
    chrome_options = Options()
    # chrome_options.add_argument("--headless") # 브라우저 창 숨기기 (원하면 주석 해제)

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        print(f"채널 접속 중... 목표: {target_count}개")
        driver.get(channel_url)
        time.sleep(2) 

        body = driver.find_element(By.TAG_NAME, "body")
        
        # 40개가 로딩될 때까지 스크롤 반복
        while True:
            titles = driver.find_elements(By.CSS_SELECTOR, "a#video-title-link")
            
            if len(titles) >= target_count:
                break
            
            body.send_keys(Keys.END)
            time.sleep(1.5)

        # 40개 추출 및 파일 저장
        print(f"링크 추출 및 '{file_name}' 저장 시작...")
        
        with open(file_name, "w", encoding="utf-8") as f:
            for i in range(target_count):
                # 안전 장치: 혹시라도 40개보다 적게 로딩되었을 경우 에러 방지
                if i >= len(titles): 
                    break
                    
                link = titles[i].get_attribute("href")
                f.write(link + "\n") # 파일에 쓰기 (줄바꿈 포함)
                print(f"[{i+1}/{target_count}] 저장: {link}")

        print(f"\n완료! '{file_name}' 파일을 확인해주세요.")

    except Exception as e:
        print(f"에러 발생: {e}")
    finally:
        driver.quit()

# 실행
target_url = "https://www.youtube.com/@%EC%97%B0%EC%95%A0%EC%96%B8%EC%96%B4/videos"
crawl_and_save(target_url, 40, "url_list.txt")
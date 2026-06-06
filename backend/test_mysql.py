# 数据库连接测试脚本
import os
import mysql.connector
from dotenv import load_dotenv

load_dotenv()

try:
    conn = mysql.connector.connect(
        host=os.getenv("MYSQL_HOST"),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DB")
    )
    print("✅ MySQL 连接成功！")
    conn.close()
except Exception as e:
    print(f"❌ 连接失败：{e}")
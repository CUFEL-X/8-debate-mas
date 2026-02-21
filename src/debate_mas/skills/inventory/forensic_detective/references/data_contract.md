# 数据契约 (Data Contract)

## etf_basic 表
| 列名 | 必选 | 说明 | 示例 |
| :--- | :--- | :--- | :--- |
| code | Yes | 证券代码 | 510300 |
| cname | Yes | 中文简称 | 沪深300ETF |
| mgt_fee | Yes | 管理费率 | 0.50 (无需百分号，或含%) |
| setup_date | Yes | 成立日期 | 20120504 或 2012-05-04 |

## csrc 表 (证监会公告)
| 列名 | 必选 | 说明 |
| :--- | :--- | :--- |
| title | Yes | 公告标题 |
| content | No | 正文内容 |
| date | Yes | 发布日期 |
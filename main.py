import matplotlib
matplotlib.use("Agg")
import requests
import feedparser
import json
import re
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from collections import Counter

# ======================
# 配置
# ======================

JOURNALS = ["Journal of Geodesy", "GPS Solutions"]

KEYWORDS = [
    "GNSS", "GPS", "Galileo", "BeiDou",
    "PPP", "RTK", "ionosphere", "troposphere",
    "orbit", "clock", "LEO"
]

CATEGORIES = {
    "PPP": ["ppp"],
    "RTK": ["rtk"],
    "Orbit": ["orbit"],
    "Clock": ["clock"],
    "Ionosphere": ["ionosphere", "tec"],
    "Troposphere": ["troposphere", "ztd"],
    "LEO": ["leo"],
    "Multi-GNSS": ["galileo", "beidou", "bds"]
}

# ======================
# 工具函数
# ======================

def clean_abstract(text):
    if not text:
        return ""
    return re.sub("<.*?>", "", text)

def extract_keywords(text, top_k=5):
    if not text:
        return []

    try:
        vec = TfidfVectorizer(stop_words="english", max_features=50)
        X = vec.fit_transform([text])
        words = vec.get_feature_names_out()
        scores = X.toarray()[0]

        pairs = sorted(zip(words, scores), key=lambda x: x[1], reverse=True)
        return [w for w, _ in pairs[:top_k]]
    except:
        return []

def classify(text):
    text = text.lower()
    tags = []
    for k, v in CATEGORIES.items():
        if any(word in text for word in v):
            tags.append(k)
    return tags if tags else ["Other"]

# ======================
# 数据抓取
# ======================

def fetch_crossref(days=7):
    url = "https://api.crossref.org/works"

    from_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

    papers = []

    for journal in JOURNALS:
        params = {
            "query": "GNSS OR GPS OR Galileo OR BeiDou OR PPP OR RTK",
            "rows": 50,
            "filter": f"from-pub-date:{from_date},container-title:{journal}",
            "sort": "published",
            "order": "desc"
        }

        r = requests.get(url, params=params)
        items = r.json()["message"]["items"]

        for it in items:
            papers.append({
                "title": it.get("title", [""])[0],
                "authors": ", ".join([a.get("family","") for a in it.get("author",[])]),
                "year": it.get("issued", {}).get("date-parts", [[None]])[0][0],
                "journal": journal,
                "url": it.get("URL"),
                "abstract": clean_abstract(it.get("abstract",""))
            })

    return papers

def fetch_arxiv(days=7):
    query = quote("gnss gps beidou galileo ppp rtk")

    url = f"http://export.arxiv.org/api/query?search_query={query}&max_results=50"

    feed = feedparser.parse(url)

    papers = []

    for e in feed.entries:
        papers.append({
            "title": e.title,
            "authors": ", ".join([a.name for a in e.authors]) if hasattr(e, "authors") else "",
            "year": e.published[:4],
            "journal": "arXiv",
            "url": e.link,
            "abstract": e.summary
        })

    return papers

# ======================
# 可视化
# ======================

def plot_trend(papers):
    years = [p["year"] for p in papers if p["year"]]
    counter = Counter(years)

    x = sorted(counter.keys())
    y = [counter[i] for i in x]

    plt.figure()
    plt.plot(x, y)
    plt.xlabel("Year")
    plt.ylabel("Papers")
    plt.title("GNSS Research Trend")
    plt.savefig("trend.png")

# ======================
# README生成
# ======================

def generate_readme(papers):
    md = "# 📡 GNSS Literature Hub\n\n"
    md += "Automatically updated GNSS papers.\n\n"
    md += "## 📈 Trend\n\n![trend](trend.png)\n\n"

    md += "## 📚 Latest Papers\n\n"

    for p in papers[:30]:
        kw = ", ".join(p.get("keywords", []))
        ab = p.get("abstract", "")[:300]

        md += f"### {p['title']}\n"
        md += f"- {p['authors']} ({p['year']})\n"
        md += f"- *{p['journal']}*\n"
        md += f"- **Keywords:** {kw}\n"
        md += f"- **Abstract:** {ab}...\n"
        md += f"- [Link]({p['url']})\n\n"

    with open("README.md", "w", encoding="utf-8") as f:
        f.write(md)

# ======================
# 主流程
# ======================

def main():
    papers = []
    papers += fetch_crossref()
    papers += fetch_arxiv()

    # 去重
    seen = set()
    unique = []
    for p in papers:
        if p["title"] not in seen:
            seen.add(p["title"])
            unique.append(p)

    # 处理
    for p in unique:
        text = p["title"] + " " + p["abstract"]
        p["keywords"] = extract_keywords(text)
        p["tags"] = classify(text)

    # 保存
    with open("papers.json", "w") as f:
        json.dump(unique, f, indent=2)

    plot_trend(unique)
    generate_readme(unique)

if __name__ == "__main__":
    main()

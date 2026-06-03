/**
 * PIA (Philippine Information Agency) News Scraper
 *
 * Scrapes pia.gov.ph for articles containing "South China Sea" or "West Philippine Sea"
 * Uses Playwright to bypass Cloudflare protection
 *
 * Usage: node pia_scraper.js
 * Output: pia_WPS_data.json
 */

const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

// Configuration
const CONFIG = {
    keywords: ['West Philippine Sea', 'South China Sea'],
    dateFrom: '2022-07-01',
    dateTo: '2026-05-31',
    outputFile: path.join(__dirname, 'pia_WPS_data.json'),
    maxPagesPerKeyword: 500, // safety limit
    requestDelay: 1500, // ms between requests to be polite
};

// Known WPS/SCS related DFA article URLs (collected manually)
const KNOWN_DFA_ARTICLES = [
    'https://dfa.gov.ph/dfa-news/news-from-our-foreign-service-postsupdate/10020-ph-embassy-discusses-west-philippine-sea-victory-at-philippine-school-doha',
];

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Extract JSON-LD data from page
 */
async function extractJsonLd(page) {
    return await page.evaluate(() => {
        const scripts = document.querySelectorAll('script[type="application/ld+json"]');
        for (const script of scripts) {
            try {
                const data = JSON.parse(script.textContent);
                if (data['@graph']) {
                    const article = data['@graph'].find(item => item['@type'] === 'Article');
                    if (article) return article;
                }
                return data;
            } catch (e) { /* skip invalid JSON */ }
        }
        return null;
    });
}

/**
 * Extract article content from page
 */
async function extractArticleContent(page) {
    return await page.evaluate(() => {
        // Try different content selectors
        const contentSelectors = [
            '.w-post-elm.post_content',
            '.post-content',
            '.entry-content',
            'article .content',
            'article',
        ];

        for (const selector of contentSelectors) {
            const el = document.querySelector(selector);
            if (el) {
                // Get all paragraphs
                const paragraphs = el.querySelectorAll('p');
                if (paragraphs.length > 0) {
                    return Array.from(paragraphs)
                        .map(p => p.textContent.trim())
                        .filter(text => text.length > 0)
                        .join('\n\n');
                }
                return el.textContent.trim();
            }
        }
        return '';
    });
}

/**
 * Search PIA for articles matching a keyword, with pagination
 */
async function searchPIA(browser, keyword, collectedUrls) {
    const context = await browser.newContext({
        userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    });

    const results = [];
    let pageNum = 0;
    let hasMore = true;

    console.log(`\n=== Searching PIA for: "${keyword}" ===`);

    while (hasMore && pageNum < CONFIG.maxPagesPerKeyword) {
        const start = pageNum * 10; // PIA uses start=N for pagination
        const searchUrl = `https://pia.gov.ph/?s=${encodeURIComponent(keyword)}&st=Blog&start=${start}`;

        const page = await context.newPage();
        try {
            await page.goto(searchUrl, { waitUntil: 'domcontentloaded', timeout: 30000 });
            await sleep(1000); // let JS render

            // Find article links
            const articles = await page.$$eval('a[href*="/news/"], a[href*="/regions/"]', els => {
                return els
                    .filter(a => {
                        const href = a.href || '';
                        // Only get article links (not category listings)
                        return (href.includes('/news/') || href.includes('/regions/')) &&
                               !href.endsWith('/news/') && !href.endsWith('/regions/');
                    })
                    .map(a => ({
                        url: a.href,
                        text: a.textContent.trim()
                    }))
                    .filter((item, index, self) =>
                        // Remove duplicates
                        index === self.findIndex(t => t.url === item.url)
                    );
            });

            // Also get search result count / total
            const resultCountText = await page.evaluate(() => {
                const el = document.querySelector('.search-result-count, .results-count, .page-title span, h1');
                return el ? el.textContent : '';
            });

            if (articles.length === 0) {
                console.log(`  Page ${pageNum}: No articles found, stopping.`);
                hasMore = false;
            } else {
                // Filter out already collected URLs
                const newArticles = articles.filter(a => !collectedUrls.has(a.url));
                newArticles.forEach(a => collectedUrls.add(a.url));

                results.push(...articles);
                console.log(`  Page ${pageNum + 1}: Found ${articles.length} articles (${newArticles.length} new). Total: ${results.length}`);

                // Check if this page had fewer than 10 results (last page)
                if (articles.length < 10) {
                    hasMore = false;
                    console.log(`  Less than 10 results on this page, assuming last page.`);
                }

                pageNum++;
            }

        } catch (err) {
            console.error(`  Error on page ${pageNum}: ${err.message.slice(0, 100)}`);
            hasMore = false;
        }
        await page.close();
    }

    await context.close();
    return results;
}

/**
 * Scrape a single article page for full content
 */
async function scrapeArticle(browser, articleInfo) {
    const context = await browser.newContext({
        userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    });
    const page = await context.newPage();

    try {
        await page.goto(articleInfo.url, { waitUntil: 'domcontentloaded', timeout: 30000 });
        await sleep(1000);

        // Extract JSON-LD
        const jsonld = await extractJsonLd(page);

        // Extract full text content
        const content = await extractArticleContent(page);

        const title = jsonld?.headline || await page.title() || articleInfo.text;
        const datePublished = jsonld?.datePublished || '';
        const author = jsonld?.author?.name || '';
        const keywords = jsonld?.keywords || [];
        const wordCount = jsonld?.wordCount || 0;
        const category = jsonld?.articleSection || [];

        // Check if within date range
        let isInRange = true;
        if (datePublished) {
            const pubDate = new Date(datePublished);
            const fromDate = new Date(CONFIG.dateFrom);
            const toDate = new Date(CONFIG.dateTo);
            isInRange = pubDate >= fromDate && pubDate <= toDate;
        }

        return {
            title: title.replace(' - Philippine Information Agency', '').trim(),
            url: articleInfo.url,
            date: datePublished,
            author: author,
            content: content,
            keywords: Array.isArray(keywords) ? keywords.join(', ') : keywords,
            wordCount: wordCount,
            category: Array.isArray(category) ? category.join(', ') : category,
            source: 'PIA',
            country: 'Philippines',
            inDateRange: isInRange,
        };

    } catch (err) {
        console.error(`  Error scraping ${articleInfo.url}: ${err.message.slice(0, 100)}`);
        return null;
    } finally {
        await page.close();
        await context.close();
    }
}

/**
 * Main function
 */
async function main() {
    console.log('=== PIA News Scraper ===');
    console.log(`Date range: ${CONFIG.dateFrom} to ${CONFIG.dateTo}`);
    console.log(`Keywords: ${CONFIG.keywords.join(', ')}`);

    const browser = await chromium.launch({
        headless: true,
        args: ['--no-sandbox', '--disable-blink-features=AutomationControlled']
    });

    try {
        // Step 1: Search for articles with each keyword
        const collectedUrls = new Set();
        let allSearchResults = [];

        for (const keyword of CONFIG.keywords) {
            const results = await searchPIA(browser, keyword, collectedUrls);
            allSearchResults.push(...results);
            await sleep(2000);
        }

        // Remove duplicates across keyword searches
        const uniqueResults = [];
        const seenUrls = new Set();
        for (const result of allSearchResults) {
            if (!seenUrls.has(result.url)) {
                seenUrls.add(result.url);
                uniqueResults.push(result);
            }
        }

        console.log(`\n=== Found ${uniqueResults.length} unique articles ===`);

        // Step 2: Scrape each article
        const articles = [];
        for (let i = 0; i < uniqueResults.length; i++) {
            const result = uniqueResults[i];
            console.log(`\n[${i + 1}/${uniqueResults.length}] ${result.text.substring(0, 60)}...`);

            const article = await scrapeArticle(browser, result);
            if (article) {
                console.log(`  Date: ${article.date.substring(0, 10) || 'N/A'}`);
                console.log(`  Content: ${article.content.substring(0, 50).trim()}...`);
                console.log(`  In range: ${article.inDateRange}`);
                articles.push(article);
            }

            // Polite delay
            await sleep(CONFIG.requestDelay);
        }

        // Step 3: Save results
        const output = {
            config: CONFIG,
            scrapedAt: new Date().toISOString(),
            totalFound: uniqueResults.length,
            totalScraped: articles.length,
            articles: articles,
        };

        fs.writeFileSync(CONFIG.outputFile, JSON.stringify(output, null, 2), 'utf-8');
        console.log(`\n=== Results saved to ${CONFIG.outputFile} ===`);
        console.log(`Total articles scraped: ${articles.length}`);
        console.log(`Articles in date range: ${articles.filter(a => a.inDateRange).length}`);

    } catch (err) {
        console.error('Fatal error:', err.message);
    } finally {
        await browser.close();
    }
}

main().catch(console.error);

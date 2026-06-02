/**
 * PIA (Philippine Information Agency) News Scraper - v2
 *
 * Scrapes pia.gov.ph for articles containing "South China Sea" or "West Philippine Sea"
 * Uses Playwright to bypass Cloudflare protection
 *
 * Usage: node pia_scraper_v2.js
 * Output: pia_WPS_data.json
 */

const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

const CONFIG = {
    keywords: ['West Philippine Sea', 'South China Sea'],
    dateFrom: '2022-07-01',
    dateTo: '2026-05-31',
    outputFile: path.join(__dirname, 'pia_WPS_data.json'),
    maxPagesPerKeyword: 200,
    requestDelay: 1500,
};

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Extract JSON-LD article data from page
 */
async function extractJsonLd(page) {
    return await page.evaluate(() => {
        const scripts = document.querySelectorAll('script[type="application/ld+json"]');
        for (const script of scripts) {
            try {
                const data = JSON.parse(script.textContent);
                const graph = data['@graph'] || [data];
                const article = graph.find(item => item['@type'] === 'Article');
                if (article) return article;
            } catch (e) { /* skip */ }
        }
        return null;
    });
}

/**
 * Extract article body text from page
 */
async function extractArticleContent(page) {
    return await page.evaluate(() => {
        // Main content selector for PIA's Impreza theme
        const contentEl = document.querySelector('.w-post-elm.post_content');
        if (contentEl) {
            const paragraphs = contentEl.querySelectorAll('p');
            if (paragraphs.length > 0) {
                return Array.from(paragraphs)
                    .map(p => p.textContent.trim())
                    .filter(t => t.length > 0)
                    .join('\n\n');
            }
            return contentEl.textContent.trim();
        }
        return '';
    });
}

/**
 * Search PIA and collect article URLs from main content area
 */
async function searchPIA(browser, keyword) {
    const context = await browser.newContext({
        userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    });

    const foundUrls = new Set();
    const foundArticles = [];
    let pageNum = 0;
    let hasMore = true;

    console.log(`\n=== Searching "${keyword}" ===`);

    while (hasMore && pageNum < CONFIG.maxPagesPerKeyword) {
        const start = pageNum * 10;
        const url = `https://pia.gov.ph/?s=${encodeURIComponent(keyword)}&st=Blog&start=${start}`;

        const page = await context.newPage();
        try {
            await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 30000 });
            await sleep(1500);

            // Get article URLs from the w-grid-list (search results grid)
            const articles = await page.evaluate(() => {
                const grid = document.querySelector('.w-grid-list');
                if (!grid) return [];

                const items = grid.querySelectorAll('.w-grid-item');
                const results = [];

                for (const item of items) {
                    // Find the article link (usually in post_image)
                    const link = item.querySelector('a[href*="/news/"], a[href*="/regions/"]');
                    if (!link) continue;

                    const href = link.href || '';
                    // Skip category listing pages
                    if (href.endsWith('/news/') || href.endsWith('/regions/')) continue;

                    // Title from aria-label or from the post_title element
                    let title = link.getAttribute('aria-label') || '';
                    if (!title) {
                        const titleEl = item.querySelector('.post_title, .entry-title, h2, h3, h4');
                        title = titleEl ? titleEl.textContent.trim() : '';
                    }
                    if (!title) {
                        title = link.textContent.trim();
                    }

                    results.push({ url: href, text: title || href.split('/').pop().replace(/-/g, ' ') });
                }
                return results;
            });

            if (articles.length === 0) {
                console.log(`  Page ${pageNum + 1}: No more articles found.`);
                hasMore = false;
            } else {
                let newCount = 0;
                for (const article of articles) {
                    if (!foundUrls.has(article.url)) {
                        foundUrls.add(article.url);
                        foundArticles.push(article);
                        newCount++;
                    }
                }
                console.log(`  Page ${pageNum + 1}: ${articles.length} links, ${newCount} new (total: ${foundArticles.length})`);

                // If fewer than 10 articles, this is likely the last page
                if (articles.length < 10) {
                    hasMore = false;
                }
                pageNum++;
            }

        } catch (err) {
            console.error(`  Error page ${pageNum + 1}: ${err.message.slice(0, 100)}`);
            hasMore = false;
        }
        await page.close();
    }

    await context.close();
    return foundArticles;
}

/**
 * Scrape full content from a single article page
 */
async function scrapeArticle(browser, articleInfo) {
    const context = await browser.newContext({
        userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    });
    const page = await context.newPage();

    try {
        await page.goto(articleInfo.url, { waitUntil: 'domcontentloaded', timeout: 30000 });
        await sleep(1500);

        const jsonld = await extractJsonLd(page);
        const content = await extractArticleContent(page);
        const title = articleInfo.text || jsonld?.headline || await page.title() || '';

        // Date from JSON-LD or HTML
        let datePublished = jsonld?.datePublished || '';
        if (!datePublished) {
            datePublished = await page.evaluate(() => {
                const timeEl = document.querySelector('time[datetime]');
                return timeEl ? timeEl.getAttribute('datetime') : '';
            });
        }

        const keywords = jsonld?.keywords || [];
        const wordCount = jsonld?.wordCount || content.length;
        const category = jsonld?.articleSection || [];

        // Check date range
        let inRange = true;
        if (datePublished) {
            const pubDate = new Date(datePublished);
            inRange = pubDate >= new Date(CONFIG.dateFrom) && pubDate <= new Date(CONFIG.dateTo);
        }

        return {
            title: title.replace(/\s*-\s*Philippine Information Agency\s*$/i, '').trim(),
            url: articleInfo.url,
            date: datePublished,
            content: content.substring(0, 50000), // limit length
            keywords: Array.isArray(keywords) ? keywords.join('; ') : String(keywords),
            wordCount: wordCount,
            category: Array.isArray(category) ? category.join('; ') : String(category),
            source: 'PIA',
            country: 'Philippines',
            inDateRange: inRange,
        };

    } catch (err) {
        console.error(`  Error scraping: ${err.message.slice(0, 100)}`);
        return null;
    } finally {
        await page.close();
        await context.close();
    }
}

async function main() {
    console.log('=== PIA News Scraper v2 ===');
    console.log(`Date: ${CONFIG.dateFrom} ~ ${CONFIG.dateTo}`);

    const browser = await chromium.launch({
        headless: true,
        args: ['--no-sandbox', '--disable-blink-features=AutomationControlled']
    });

    try {
        // Step 1: Collect all article URLs across keywords
        const allUrls = new Set();
        let allArticles = [];

        for (const keyword of CONFIG.keywords) {
            const results = await searchPIA(browser, keyword);
            for (const r of results) {
                if (!allUrls.has(r.url)) {
                    allUrls.add(r.url);
                    allArticles.push(r);
                }
            }
            await sleep(2000);
        }

        console.log(`\n=== Total unique articles found: ${allArticles.length} ===`);

        // Step 2: Scrape full content
        const scraped = [];
        for (let i = 0; i < allArticles.length; i++) {
            const a = allArticles[i];
            const shortTitle = a.text.substring(0, 60);
            process.stdout.write(`[${i+1}/${allArticles.length}] ${shortTitle}... `);

            const article = await scrapeArticle(browser, a);
            if (article) {
                scraped.push(article);
                console.log(`✓ (${article.date.substring(0,10) || 'no date'})`);
            } else {
                console.log('✗');
            }

            await sleep(CONFIG.requestDelay);
        }

        // Step 3: Save
        const output = {
            config: CONFIG,
            scrapedAt: new Date().toISOString(),
            totalFound: allArticles.length,
            totalScraped: scraped.length,
            inRangeCount: scraped.filter(a => a.inDateRange).length,
            articles: scraped,
        };

        fs.writeFileSync(CONFIG.outputFile, JSON.stringify(output, null, 2), 'utf-8');
        console.log(`\nSaved to ${CONFIG.outputFile}`);
        console.log(`Scraped: ${scraped.length} total, ${output.inRangeCount} in date range`);

    } catch (err) {
        console.error('Fatal:', err.message);
    } finally {
        await browser.close();
    }
}

main().catch(console.error);

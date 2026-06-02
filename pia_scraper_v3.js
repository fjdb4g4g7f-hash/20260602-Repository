/**
 * PIA (Philippine Information Agency) News Scraper - v3
 *
 * Uses WordPress REST API to find articles about WPS/SCS.
 * Much more efficient than HTML scraping.
 *
 * Usage: node pia_scraper_v3.js
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
    perPage: 100,
};

function sleep(ms) {
    return new Promise(r => setTimeout(r, ms));
}

/**
 * Fetch all posts from WP REST API for a keyword search
 */
async function fetchAllPosts(browser, keyword) {
    const context = await browser.newContext({
        userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    });
    const page = await context.newPage();

    const allPosts = [];
    let pageNum = 1;
    let hasMore = true;

    console.log(`\n=== Fetching: "${keyword}" ===`);

    while (hasMore) {
        const url = `https://pia.gov.ph/wp-json/wp/v2/posts?search=${encodeURIComponent(keyword)}&per_page=${CONFIG.perPage}&page=${pageNum}&_fields=id,title,date,link,content,excerpt,yoast_head_json,tags`;
        await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 30000 });
        await sleep(500);

        const result = await page.evaluate(() => {
            try {
                const data = JSON.parse(document.body.textContent);
                if (!Array.isArray(data)) return { posts: [], error: 'Not an array' };
                return { posts: data.map(p => ({
                    id: p.id,
                    title: p.title?.rendered || '',
                    date: p.date || '',
                    link: p.link || '',
                    content: p.content?.rendered || '',
                    excerpt: p.excerpt?.rendered || '',
                    yoast: p.yoast_head_json || null,
                }))};
            } catch(e) {
                return { posts: [], error: e.message };
            }
        });

        if (result.posts.length === 0) {
            hasMore = false;
        } else {
            allPosts.push(...result.posts);
            console.log(`  Page ${pageNum}: ${result.posts.length} posts (total: ${allPosts.length})`);
            if (result.posts.length < CONFIG.perPage) {
                hasMore = false;
            }
            pageNum++;
        }
    }

    await page.close();
    await context.close();
    return allPosts;
}

/**
 * Check if an article is relevant to WPS/SCS
 */
function isRelevant(post) {
    const title = (post.title || '').toLowerCase();
    const content = (post.content || '').toLowerCase();
    const excerpt = (post.excerpt || '').toLowerCase();

    // Primary keywords (must be in title or content)
    const primaryKeywords = [
        'west philippine sea', 'wps',
        'south china sea',
        'arbitral award',
        'pagtindig para sa west philippine sea',
        'defending west philippine',
        'maritime security',
    ];

    // Check title first (stronger signal)
    for (const kw of primaryKeywords) {
        if (title.includes(kw)) return true;
    }

    // For content, require at least one main keyword
    const mainKeywords = ['west philippine sea', 'south china sea', 'wps'];
    for (const kw of mainKeywords) {
        if (content.includes(kw)) return true;
    }

    return false;
}

/**
 * Strip HTML tags and get clean text
 */
function stripHtml(html) {
    return html
        .replace(/<[^>]*>/g, ' ')
        .replace(/&amp;/g, '&')
        .replace(/&nbsp;/g, ' ')
        .replace(/&#8217;/g, "'")
        .replace(/&#8216;/g, "'")
        .replace(/&#8220;/g, '"')
        .replace(/&#8221;/g, '"')
        .replace(/&#038;/g, '&')
        .replace(/\s+/g, ' ')
        .trim();
}

async function main() {
    console.log('=== PIA News Scraper v3 (WP REST API) ===');
    console.log(`Date: ${CONFIG.dateFrom} ~ ${CONFIG.dateTo}`);

    const browser = await chromium.launch({
        headless: true,
        args: ['--no-sandbox', '--disable-blink-features=AutomationControlled']
    });

    try {
        // Step 1: Fetch all posts for each keyword
        let allPosts = [];
        const seenIds = new Set();

        for (const keyword of CONFIG.keywords) {
            const posts = await fetchAllPosts(browser, keyword);
            for (const post of posts) {
                if (!seenIds.has(post.id)) {
                    seenIds.add(post.id);
                    allPosts.push(post);
                }
            }
            console.log(`  -> ${posts.length} total, ${allPosts.length} unique across all keywords`);
        }

        console.log(`\n=== Total unique posts: ${allPosts.length} ===`);

        // Step 2: Filter by relevance
        const relevant = allPosts.filter(isRelevant);
        console.log(`Relevant (WPS/SCS keywords): ${relevant.length}`);

        // Step 3: Filter by date range
        const fromDate = new Date(CONFIG.dateFrom);
        const toDate = new Date(CONFIG.dateTo);
        const inRange = relevant.filter(p => {
            const d = new Date(p.date);
            return d >= fromDate && d <= toDate;
        });
        console.log(`In date range ${CONFIG.dateFrom} ~ ${CONFIG.dateTo}: ${inRange.length}`);

        // Step 4: Build output
        const articles = inRange.map(post => ({
            title: stripHtml(post.title),
            url: post.link,
            date: post.date,
            content: stripHtml(post.content),
            excerpt: stripHtml(post.excerpt),
            source: 'PIA',
            country: 'Philippines',
        }));

        // Step 5: Deduplicate by title/date
        const deduped = [];
        const seen = new Set();
        for (const a of articles) {
            const key = `${a.title}|${a.date.substring(0, 10)}`;
            if (!seen.has(key)) {
                seen.add(key);
                deduped.push(a);
            }
        }
        console.log(`After dedup: ${deduped.length}`);

        // Step 6: Save
        const output = {
            config: CONFIG,
            scrapedAt: new Date().toISOString(),
            totalApiResults: allPosts.length,
            totalRelevant: relevant.length,
            totalInRange: inRange.length,
            finalCount: deduped.length,
            articles: deduped,
        };

        fs.writeFileSync(CONFIG.outputFile, JSON.stringify(output, null, 2), 'utf-8');
        console.log(`\nSaved to ${CONFIG.outputFile}`);
        console.log(`Final article count: ${deduped.length}`);

        // Show year distribution
        const yearDist = {};
        for (const a of deduped) {
            const year = a.date.substring(0, 4);
            yearDist[year] = (yearDist[year] || 0) + 1;
        }
        console.log('Year distribution:', yearDist);

    } catch (err) {
        console.error('Fatal:', err.message);
    } finally {
        await browser.close();
    }
}

main().catch(console.error);

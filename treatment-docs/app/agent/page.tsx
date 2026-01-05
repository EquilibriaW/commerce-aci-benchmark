import { getCart, getProducts } from 'lib/shopify';
import { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Commerce API Reference v1.0',
  description: 'API Documentation for Commerce Operations'
};

export default async function AgentDashboard() {
  const [cart, products] = await Promise.all([
    getCart(),
    getProducts({})
  ]);

  const cartJson = cart ? JSON.stringify(cart, null, 2) : '{ "items": [], "total": 0 }';

  // Group products by category (simulated modules)
  const apparelProducts = products.filter(p =>
    p.handle.includes('shirt') || p.handle.includes('hoodie')
  );
  const accessoryProducts = products.filter(p =>
    p.handle.includes('cup') || p.handle.includes('bag') || p.handle.includes('cap')
  );
  const otherProducts = products.filter(p =>
    !apparelProducts.includes(p) && !accessoryProducts.includes(p)
  );

  return (
    <html lang="en">
      <head>
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <meta name="robots" content="noindex" />
        <title>Commerce API Reference v1.0</title>
        <style dangerouslySetInnerHTML={{ __html: `
          * { box-sizing: border-box; margin: 0; padding: 0; }
          body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: #fcfcfc;
            color: #333;
            line-height: 1.6;
          }
          .doc-container {
            display: flex;
            min-height: 100vh;
          }
          .sidebar {
            width: 250px;
            background: #f3f4f6;
            border-right: 1px solid #e5e7eb;
            padding: 20px 0;
            position: fixed;
            height: 100vh;
            overflow-y: auto;
          }
          .sidebar-header {
            padding: 0 20px 15px;
            border-bottom: 1px solid #e5e7eb;
            margin-bottom: 15px;
          }
          .sidebar-header h1 {
            font-family: 'Georgia', serif;
            font-size: 16px;
            color: #1a1a1a;
            margin-bottom: 5px;
          }
          .sidebar-header .version {
            font-size: 12px;
            color: #6b7280;
          }
          .sidebar-nav {
            padding: 0 15px;
          }
          .sidebar-nav a {
            display: block;
            padding: 8px 12px;
            color: #2980b9;
            text-decoration: none;
            font-size: 14px;
            border-radius: 4px;
          }
          .sidebar-nav a:hover {
            background: #e5e7eb;
            text-decoration: underline;
          }
          .sidebar-nav a.active {
            background: #2980b9;
            color: white;
          }
          .nav-section {
            margin-bottom: 20px;
          }
          .nav-section-title {
            font-size: 11px;
            text-transform: uppercase;
            color: #6b7280;
            padding: 0 12px;
            margin-bottom: 8px;
            letter-spacing: 0.5px;
          }
          .main-content {
            margin-left: 250px;
            padding: 30px 40px;
            max-width: 900px;
            flex: 1;
          }
          h1.page-title {
            font-family: 'Georgia', serif;
            font-size: 28px;
            color: #1a1a1a;
            border-bottom: 2px solid #e5e7eb;
            padding-bottom: 15px;
            margin-bottom: 25px;
          }
          h2.section-title {
            font-family: 'Georgia', serif;
            font-size: 20px;
            color: #1a1a1a;
            margin: 30px 0 15px;
            padding-bottom: 8px;
            border-bottom: 1px solid #e5e7eb;
          }
          h3.module-title {
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 16px;
            color: #2980b9;
            margin: 20px 0 10px;
          }
          .session-var {
            background: #f8f9fa;
            border: 1px solid #e5e7eb;
            border-left: 4px solid #2980b9;
            padding: 15px;
            margin: 15px 0;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 13px;
          }
          .session-var .var-name {
            color: #d73a49;
          }
          .session-var pre {
            margin-top: 10px;
            background: #f1f3f5;
            padding: 12px;
            overflow-x: auto;
            font-size: 12px;
            border-radius: 4px;
          }
          .class-list {
            list-style: none;
            margin: 10px 0;
          }
          .class-item {
            padding: 12px 15px;
            border: 1px solid #e5e7eb;
            margin-bottom: 8px;
            border-radius: 4px;
            background: white;
          }
          .class-item:hover {
            border-color: #2980b9;
            background: #f8fafc;
          }
          .class-name {
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 14px;
            color: #2980b9;
            text-decoration: none;
          }
          .class-name:hover {
            text-decoration: underline;
          }
          .class-name .keyword {
            color: #d73a49;
          }
          .class-summary {
            font-size: 13px;
            color: #6b7280;
            margin-top: 5px;
            font-style: italic;
          }
          .class-meta {
            font-size: 12px;
            color: #9ca3af;
            margin-top: 8px;
          }
          .class-meta span {
            margin-right: 15px;
          }
          .badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 11px;
            font-weight: 500;
          }
          .badge-stock {
            background: #d4edda;
            color: #155724;
          }
          .badge-price {
            background: #e2e3e5;
            color: #383d41;
          }
          .footer {
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #e5e7eb;
            font-size: 12px;
            color: #9ca3af;
          }
        `}} />
      </head>
      <body>
        <div className="doc-container">
          <aside className="sidebar">
            <div className="sidebar-header">
              <h1>Commerce API</h1>
              <div className="version">v1.0 Reference</div>
            </div>
            <nav className="sidebar-nav">
              <div className="nav-section">
                <div className="nav-section-title">Navigation</div>
                <a href="/agent" className="active">Module Index</a>
                <a href="/llms.txt">llms.txt</a>
              </div>
              <div className="nav-section">
                <div className="nav-section-title">Modules</div>
                <a href="#products-apparel">products.apparel</a>
                <a href="#products-accessories">products.accessories</a>
                <a href="#products-other">products.other</a>
              </div>
              <div className="nav-section">
                <div className="nav-section-title">Session</div>
                <a href="#session-cart">session.cart</a>
              </div>
            </nav>
          </aside>

          <main className="main-content">
            <h1 className="page-title">Commerce API Reference v1.0</h1>

            <section id="session-cart">
              <h2 className="section-title">Session Variables</h2>
              <div className="session-var" data-agent-id="state:cart">
                <span className="var-name">current_session.cart</span> =
                <pre>{cartJson}</pre>
              </div>
            </section>

            <section id="products-apparel">
              <h2 className="section-title">Module Index</h2>

              <h3 className="module-title">module: products.apparel</h3>
              <ul className="class-list">
                {apparelProducts.map(product => (
                  <li key={product.id} className="class-item" data-agent-id={`product:${product.handle}`}>
                    <a href={`/agent/product/${product.handle}`} className="class-name" data-agent-id={`nav:product:${product.handle}`}>
                      <span className="keyword">class</span> {product.title.replace(/\s+/g, '')}
                    </a>
                    <div className="class-summary">{product.description || 'Product instance'}</div>
                    <div className="class-meta">
                      <span className="badge badge-price">${product.priceRange.minVariantPrice.amount}</span>
                      <span className="badge badge-stock">{product.availableForSale ? 'IN_STOCK' : 'OUT_OF_STOCK'}</span>
                      <span>handle="{product.handle}"</span>
                    </div>
                  </li>
                ))}
              </ul>
            </section>

            <section id="products-accessories">
              <h3 className="module-title">module: products.accessories</h3>
              <ul className="class-list">
                {accessoryProducts.map(product => (
                  <li key={product.id} className="class-item" data-agent-id={`product:${product.handle}`}>
                    <a href={`/agent/product/${product.handle}`} className="class-name" data-agent-id={`nav:product:${product.handle}`}>
                      <span className="keyword">class</span> {product.title.replace(/\s+/g, '')}
                    </a>
                    <div className="class-summary">{product.description || 'Product instance'}</div>
                    <div className="class-meta">
                      <span className="badge badge-price">${product.priceRange.minVariantPrice.amount}</span>
                      <span className="badge badge-stock">{product.availableForSale ? 'IN_STOCK' : 'OUT_OF_STOCK'}</span>
                      <span>handle="{product.handle}"</span>
                    </div>
                  </li>
                ))}
              </ul>
            </section>

            {otherProducts.length > 0 && (
              <section id="products-other">
                <h3 className="module-title">module: products.other</h3>
                <ul className="class-list">
                  {otherProducts.map(product => (
                    <li key={product.id} className="class-item" data-agent-id={`product:${product.handle}`}>
                      <a href={`/agent/product/${product.handle}`} className="class-name" data-agent-id={`nav:product:${product.handle}`}>
                        <span className="keyword">class</span> {product.title.replace(/\s+/g, '')}
                      </a>
                      <div className="class-summary">{product.description || 'Product instance'}</div>
                      <div className="class-meta">
                        <span className="badge badge-price">${product.priceRange.minVariantPrice.amount}</span>
                        <span className="badge badge-stock">{product.availableForSale ? 'IN_STOCK' : 'OUT_OF_STOCK'}</span>
                        <span>handle="{product.handle}"</span>
                      </div>
                    </li>
                  ))}
                </ul>
              </section>
            )}

            <div className="footer">
              <div>Commerce API Reference v1.0 | Generated: {new Date().toISOString()}</div>
            </div>
          </main>
        </div>
      </body>
    </html>
  );
}

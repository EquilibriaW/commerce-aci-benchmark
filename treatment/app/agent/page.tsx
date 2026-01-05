import { getCart, getProducts } from 'lib/shopify';
import { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'ACI // Agent Computer Interface',
  description: 'System Terminal for Agent Operations'
};

export default async function AgentDashboard() {
  const [cart, products] = await Promise.all([
    getCart(),
    getProducts({})
  ]);

  const cartJson = cart ? JSON.stringify(cart, null, 2) : '{ "status": "empty", "items": [] }';

  return (
    <html lang="en">
      <head>
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <meta name="robots" content="noindex" />
        <title>ACI // Agent Computer Interface</title>
        <style dangerouslySetInnerHTML={{ __html: `
          * { box-sizing: border-box; }
          body {
            font-family: 'Courier New', Courier, monospace;
            background: #ffffff;
            color: #000000;
            margin: 0;
            padding: 20px;
            font-size: 14px;
            line-height: 1.4;
          }
          .header {
            border: 2px solid #000;
            padding: 10px 15px;
            margin-bottom: 20px;
            background: #f0f0f0;
          }
          .header h1 {
            margin: 0;
            font-size: 18px;
            font-weight: bold;
          }
          .header .status {
            color: #008000;
            font-weight: bold;
          }
          nav {
            border: 1px solid #000;
            padding: 8px 12px;
            margin-bottom: 20px;
            background: #fafafa;
          }
          nav a {
            color: #0000cc;
            text-decoration: underline;
            margin-right: 20px;
          }
          nav a:hover { color: #000099; }
          .section {
            border: 1px solid #000;
            margin-bottom: 20px;
          }
          .section-header {
            background: #000;
            color: #fff;
            padding: 6px 12px;
            font-weight: bold;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
          }
          .section-body {
            padding: 12px;
          }
          pre {
            background: #f5f5f5;
            border: 1px solid #ccc;
            padding: 12px;
            overflow-x: auto;
            margin: 0;
            font-size: 12px;
            max-height: 300px;
            overflow-y: auto;
          }
          table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
          }
          th, td {
            border: 1px solid #000;
            padding: 8px 10px;
            text-align: left;
          }
          th {
            background: #e0e0e0;
            font-weight: bold;
            text-transform: uppercase;
            font-size: 11px;
            letter-spacing: 0.5px;
          }
          tr:nth-child(even) { background: #f9f9f9; }
          tr:hover { background: #ffffcc; }
          .btn-select {
            display: inline-block;
            background: #000;
            color: #fff;
            padding: 4px 12px;
            text-decoration: none;
            font-weight: bold;
            font-size: 12px;
            border: none;
          }
          .btn-select:hover {
            background: #333;
            color: #fff;
          }
          .footer {
            border-top: 2px solid #000;
            padding-top: 10px;
            margin-top: 20px;
            font-size: 11px;
            color: #666;
          }
        `}} />
      </head>
      <body>
        <div className="header">
          <h1>ACI // Agent Computer Interface - <span className="status">SYSTEM ONLINE</span></h1>
        </div>

        <nav>
          <a href="/agent">[ DASHBOARD ]</a>
          <a href="/llms.txt">[ LLMS.TXT ]</a>
        </nav>

        <div className="section">
          <div className="section-header">Active Cart State</div>
          <div className="section-body">
            <pre data-agent-id="state:cart">{cartJson}</pre>
          </div>
        </div>

        <div className="section">
          <div className="section-header">Product Database ({products.length} records)</div>
          <div className="section-body">
            <table>
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Handle</th>
                  <th>Price</th>
                  <th>Availability</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {products.map(product => (
                  <tr key={product.id} data-agent-id={`product:${product.handle}`}>
                    <td>{product.id.slice(-8)}</td>
                    <td>
                      <a
                        href={`/agent/product/${product.handle}`}
                        data-agent-id={`nav:product:${product.handle}`}
                      >
                        {product.handle}
                      </a>
                    </td>
                    <td>${product.priceRange.minVariantPrice.amount}</td>
                    <td>{product.availableForSale ? 'IN_STOCK' : 'OUT_OF_STOCK'}</td>
                    <td>
                      <a
                        href={`/agent/product/${product.handle}`}
                        className="btn-select"
                        data-agent-id={`select:${product.handle}`}
                      >
                        [SELECT]
                      </a>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="footer">
          <div>ACI System v1.0 | Session: {new Date().toISOString()}</div>
        </div>
      </body>
    </html>
  );
}

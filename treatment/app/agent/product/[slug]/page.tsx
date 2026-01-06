import { getProduct } from 'lib/shopify';
import { cookies } from 'next/headers';
import { notFound } from 'next/navigation';

export default async function AgentProductPage({
  params
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const [product, cookieStore] = await Promise.all([
    getProduct(slug),
    cookies()
  ]);

  // Check capability mode - in parity mode, hide interactive actions
  const capability = cookieStore.get('bench_capability')?.value;
  const isParity = capability === 'parity';

  if (!product) {
    notFound();
  }

  return (
    <html lang="en">
      <head>
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <meta name="robots" content="noindex" />
        <title>ACI // {product.handle}</title>
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
            font-size: 16px;
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
            width: 25%;
          }
          tr:hover { background: #ffffcc; }
          .variant-table th {
            width: auto;
          }
          .action-box {
            border: 2px solid #000;
            padding: 15px;
            background: #ffffd0;
            margin-top: 15px;
          }
          .action-box h3 {
            margin: 0 0 10px 0;
            font-size: 14px;
            text-transform: uppercase;
          }
          .form-row {
            margin-bottom: 10px;
          }
          .form-row label {
            display: inline-block;
            width: 100px;
            font-weight: bold;
          }
          select, input[type="number"] {
            font-family: 'Courier New', Courier, monospace;
            padding: 6px 10px;
            border: 1px solid #000;
            font-size: 13px;
          }
          .btn-execute {
            display: inline-block;
            background: #000;
            color: #fff;
            padding: 10px 20px;
            text-decoration: none;
            font-weight: bold;
            font-size: 14px;
            border: 2px solid #000;
            cursor: pointer;
            font-family: 'Courier New', Courier, monospace;
            margin-top: 10px;
          }
          .btn-execute:hover {
            background: #333;
          }
          .footer {
            border-top: 2px solid #000;
            padding-top: 10px;
            margin-top: 20px;
            font-size: 11px;
            color: #666;
          }
          .footer a {
            color: #0000cc;
          }
        `}} />
      </head>
      <body>
        <div className="header">
          <h1>ACI // PRODUCT: {product.handle.toUpperCase()}</h1>
        </div>

        <nav>
          <a href="/agent">[ DASHBOARD ]</a>
          <a href="/llms.txt">[ LLMS.TXT ]</a>
        </nav>

        <div className="section" data-agent-id={`product:${product.handle}`}>
          <div className="section-header">Product Record</div>
          <div className="section-body">
            <table>
              <tbody>
                <tr>
                  <th>ID</th>
                  <td data-agent-id="product:id">{product.id}</td>
                </tr>
                <tr>
                  <th>Handle</th>
                  <td data-agent-id="product:handle">{product.handle}</td>
                </tr>
                <tr>
                  <th>Title</th>
                  <td>{product.title}</td>
                </tr>
                <tr>
                  <th>Price</th>
                  <td data-agent-id="product:price">
                    ${product.priceRange.minVariantPrice.amount} {product.priceRange.minVariantPrice.currencyCode}
                  </td>
                </tr>
                <tr>
                  <th>Status</th>
                  <td data-agent-id="product:available">
                    {product.availableForSale ? 'IN_STOCK' : 'OUT_OF_STOCK'}
                  </td>
                </tr>
                <tr>
                  <th>Description</th>
                  <td data-agent-id="product:description">{product.description}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <div className="section">
          <div className="section-header">Variant Table ({product.variants.length} variants)</div>
          <div className="section-body">
            <table className="variant-table">
              <thead>
                <tr>
                  <th>Variant ID</th>
                  <th>Option</th>
                  <th>Price</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {product.variants.map(variant => (
                  <tr key={variant.id} data-agent-id={`variant:${variant.id}`}>
                    <td>{variant.id}</td>
                    <td>{variant.title}</td>
                    <td>${variant.price.amount}</td>
                    <td>{variant.availableForSale ? 'IN_STOCK' : 'OUT_OF_STOCK'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {!isParity && (
          <div className="section">
            <div className="section-header">Execute Action</div>
            <div className="section-body">
              <div className="action-box">
                <h3>Add to Cart</h3>
                <form action="/agent/actions/add" method="POST" data-agent-id={`action:add_to_cart:${product.handle}`}>
                  <input type="hidden" name="productHandle" value={product.handle} />

                  <div className="form-row">
                    <label htmlFor="variantId">VARIANT:</label>
                    <select name="variantId" id="variantId" required>
                      {product.variants.map(variant => (
                        <option key={variant.id} value={variant.id}>
                          {variant.title} - ${variant.price.amount}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div className="form-row">
                    <label htmlFor="quantity">QTY:</label>
                    <input
                      type="number"
                      name="quantity"
                      id="quantity"
                      defaultValue={1}
                      min={1}
                      max={10}
                      required
                    />
                  </div>

                  <button type="submit" className="btn-execute" data-agent-id={`submit:add_to_cart:${product.handle}`}>
                    [ EXECUTE: ADD_TO_CART ]
                  </button>
                </form>
              </div>
            </div>
          </div>
        )}

        {isParity && (
          <div className="section">
            <div className="section-header">Actions</div>
            <div className="section-body">
              <p style={{ color: '#666', fontStyle: 'italic' }}>
                [Parity mode: Use the regular product pages to add items to cart]
              </p>
            </div>
          </div>
        )}

        <div className="footer">
          <a href="/agent">[ RETURN TO DASHBOARD ]</a>
          <div>Session: {new Date().toISOString()}</div>
        </div>
      </body>
    </html>
  );
}

import { getProduct } from 'lib/shopify';
import { cookies } from 'next/headers';
import { notFound } from 'next/navigation';
import { Metadata } from 'next';

// Dynamic metadata based on product
export async function generateMetadata({
  params
}: {
  params: Promise<{ slug: string }>;
}): Promise<Metadata> {
  const { slug } = await params;
  const product = await getProduct(slug);

  if (!product) {
    return { title: 'Product Not Found' };
  }

  return {
    title: `ACI // ${product.handle}`,
    robots: 'noindex'
  };
}

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
    <>
      <style dangerouslySetInnerHTML={{ __html: `
        .agent-product * { box-sizing: border-box; }
        .agent-product {
          font-family: 'Courier New', Courier, monospace;
          background: #ffffff;
          color: #000000;
          padding: 20px;
          font-size: 14px;
          line-height: 1.4;
        }
        .agent-product .header {
          border: 2px solid #000;
          padding: 10px 15px;
          margin-bottom: 20px;
          background: #f0f0f0;
        }
        .agent-product .header h1 {
          margin: 0;
          font-size: 16px;
          font-weight: bold;
        }
        .agent-product nav {
          border: 1px solid #000;
          padding: 8px 12px;
          margin-bottom: 20px;
          background: #fafafa;
        }
        .agent-product nav a {
          color: #0000cc;
          text-decoration: underline;
          margin-right: 20px;
        }
        .agent-product nav a:hover { color: #000099; }
        .agent-product .section {
          border: 1px solid #000;
          margin-bottom: 20px;
        }
        .agent-product .section-header {
          background: #000;
          color: #fff;
          padding: 6px 12px;
          font-weight: bold;
          font-size: 12px;
          text-transform: uppercase;
          letter-spacing: 1px;
        }
        .agent-product .section-body {
          padding: 12px;
        }
        .agent-product table {
          width: 100%;
          border-collapse: collapse;
          font-size: 13px;
        }
        .agent-product th, .agent-product td {
          border: 1px solid #000;
          padding: 8px 10px;
          text-align: left;
        }
        .agent-product th {
          background: #e0e0e0;
          font-weight: bold;
          text-transform: uppercase;
          font-size: 11px;
          letter-spacing: 0.5px;
          width: 25%;
        }
        .agent-product tr:hover { background: #ffffcc; }
        .agent-product .variant-table th {
          width: auto;
        }
        .agent-product .action-box {
          border: 2px solid #000;
          padding: 15px;
          background: #ffffd0;
          margin-top: 15px;
        }
        .agent-product .action-box h3 {
          margin: 0 0 10px 0;
          font-size: 14px;
          text-transform: uppercase;
        }
        .agent-product .form-row {
          margin-bottom: 10px;
        }
        .agent-product .form-row label {
          display: inline-block;
          width: 100px;
          font-weight: bold;
        }
        .agent-product select, .agent-product input[type="number"] {
          font-family: 'Courier New', Courier, monospace;
          padding: 6px 10px;
          border: 1px solid #000;
          font-size: 13px;
        }
        .agent-product .btn-execute {
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
        .agent-product .btn-execute:hover {
          background: #333;
        }
        .agent-product .footer {
          border-top: 2px solid #000;
          padding-top: 10px;
          margin-top: 20px;
          font-size: 11px;
          color: #666;
        }
        .agent-product .footer a {
          color: #0000cc;
        }
      `}} />

      <div className="agent-product">
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
      </div>
    </>
  );
}

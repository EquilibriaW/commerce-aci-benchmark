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
    <>
      <style dangerouslySetInnerHTML={{ __html: `
        .doc-container {
          display: flex;
          min-height: calc(100vh - 80px);
          background: #fcfcfc;
        }
        .sidebar {
          width: 250px;
          background: #f3f4f6;
          border-right: 1px solid #e5e7eb;
          padding: 20px 0;
          position: sticky;
          top: 0;
          height: calc(100vh - 80px);
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
          padding: 30px 40px;
          max-width: 900px;
          flex: 1;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
          color: #333;
          line-height: 1.6;
        }
        .breadcrumb {
          font-size: 13px;
          color: #6b7280;
          margin-bottom: 15px;
        }
        .breadcrumb a {
          color: #2980b9;
          text-decoration: none;
        }
        .breadcrumb a:hover {
          text-decoration: underline;
        }
        h1.class-title {
          font-family: 'Consolas', 'Monaco', monospace;
          font-size: 24px;
          color: #1a1a1a;
          margin-bottom: 10px;
        }
        h1.class-title .keyword {
          color: #d73a49;
        }
        h1.class-title .param {
          color: #6f42c1;
        }
        .docstring {
          background: #f8f9fa;
          border-left: 4px solid #2980b9;
          padding: 15px 20px;
          margin: 20px 0;
          font-style: italic;
          color: #555;
        }
        h2.section-title {
          font-family: 'Georgia', serif;
          font-size: 18px;
          color: #1a1a1a;
          margin: 30px 0 15px;
          padding-bottom: 8px;
          border-bottom: 1px solid #e5e7eb;
        }
        .attributes-list {
          background: #f8f9fa;
          border: 1px solid #e5e7eb;
          border-radius: 4px;
          padding: 15px 20px;
          font-family: 'Consolas', 'Monaco', monospace;
          font-size: 13px;
        }
        .attribute {
          margin: 8px 0;
        }
        .attr-name {
          color: #d73a49;
        }
        .attr-type {
          color: #6f42c1;
        }
        .attr-value {
          color: #22863a;
        }
        .variants-table {
          width: 100%;
          border-collapse: collapse;
          margin: 15px 0;
          font-size: 13px;
        }
        .variants-table th,
        .variants-table td {
          padding: 10px 12px;
          border: 1px solid #e5e7eb;
          text-align: left;
        }
        .variants-table th {
          background: #f3f4f6;
          font-weight: 600;
          font-size: 12px;
          text-transform: uppercase;
          color: #6b7280;
        }
        .variants-table tr:hover {
          background: #f8fafc;
        }
        .method-block {
          background: #1e1e1e;
          border-radius: 6px;
          padding: 20px;
          margin: 20px 0;
        }
        .method-signature {
          font-family: 'Consolas', 'Monaco', monospace;
          font-size: 14px;
          color: #9cdcfe;
          margin-bottom: 15px;
        }
        .method-signature .kw {
          color: #c586c0;
        }
        .method-signature .func {
          color: #dcdcaa;
        }
        .method-signature .prm {
          color: #9cdcfe;
        }
        .method-signature .str {
          color: #ce9178;
        }
        .form-row {
          margin: 12px 0;
        }
        .form-row label {
          display: block;
          font-family: 'Consolas', 'Monaco', monospace;
          font-size: 12px;
          color: #6a9955;
          margin-bottom: 5px;
        }
        .form-row select,
        .form-row input {
          background: #2d2d2d;
          border: 1px solid #3c3c3c;
          color: #d4d4d4;
          padding: 8px 12px;
          font-family: 'Consolas', 'Monaco', monospace;
          font-size: 13px;
          border-radius: 4px;
        }
        .execute-btn {
          display: inline-block;
          background: #2980b9;
          color: white;
          padding: 12px 24px;
          border: none;
          border-radius: 4px;
          font-family: 'Consolas', 'Monaco', monospace;
          font-size: 14px;
          cursor: pointer;
          margin-top: 15px;
        }
        .execute-btn:hover {
          background: #1f6dad;
        }
        .doc-footer {
          margin-top: 50px;
          padding-top: 20px;
          border-top: 1px solid #e5e7eb;
          font-size: 12px;
          color: #9ca3af;
        }
        .doc-footer a {
          color: #2980b9;
          text-decoration: none;
        }
        .doc-footer a:hover {
          text-decoration: underline;
        }
      `}} />

      <div className="doc-container">
        <aside className="sidebar">
          <div className="sidebar-header">
            <h1>Commerce API</h1>
            <div className="version">v1.0 Reference</div>
          </div>
          <nav className="sidebar-nav">
            <div className="nav-section">
              <div className="nav-section-title">Navigation</div>
              <a href="/agent">Module Index</a>
              <a href="/llms.txt">llms.txt</a>
            </div>
            <div className="nav-section">
              <div className="nav-section-title">This Page</div>
              <a href="#attributes">Attributes</a>
              <a href="#variants">Variants</a>
              <a href="#methods">Methods</a>
            </div>
          </nav>
        </aside>

        <main className="main-content" data-agent-id={`product:${product.handle}`}>
          <div className="breadcrumb">
            <a href="/agent">Module Index</a> &gt; products &gt; {product.handle}
          </div>

          <h1 className="class-title">
            <span className="keyword">class</span> Product(<span className="param">id</span>=&quot;{product.handle}&quot;)
          </h1>

          <div className="docstring" data-agent-id="product:description">
            &quot;{product.description || 'A product available for purchase.'}&quot;
          </div>

          <section id="attributes">
            <h2 className="section-title">Class Attributes</h2>
            <div className="attributes-list">
              <div className="attribute">
                <span className="attr-name">id</span>: <span className="attr-type">str</span> = <span className="attr-value">&quot;{product.id}&quot;</span>
              </div>
              <div className="attribute">
                <span className="attr-name">handle</span>: <span className="attr-type">str</span> = <span className="attr-value">&quot;{product.handle}&quot;</span>
              </div>
              <div className="attribute">
                <span className="attr-name">title</span>: <span className="attr-type">str</span> = <span className="attr-value">&quot;{product.title}&quot;</span>
              </div>
              <div className="attribute" data-agent-id="product:price">
                <span className="attr-name">price</span>: <span className="attr-type">Money</span> = <span className="attr-value">${product.priceRange.minVariantPrice.amount} {product.priceRange.minVariantPrice.currencyCode}</span>
              </div>
              <div className="attribute" data-agent-id="product:available">
                <span className="attr-name">available_for_sale</span>: <span className="attr-type">bool</span> = <span className="attr-value">{product.availableForSale ? 'True' : 'False'}</span>
              </div>
            </div>
          </section>

          <section id="variants">
            <h2 className="section-title">Variants</h2>
            <table className="variants-table">
              <thead>
                <tr>
                  <th>variant_id</th>
                  <th>option</th>
                  <th>price</th>
                  <th>available</th>
                </tr>
              </thead>
              <tbody>
                {product.variants.map(variant => (
                  <tr key={variant.id} data-agent-id={`variant:${variant.id}`}>
                    <td style={{ fontFamily: 'Consolas, Monaco, monospace', fontSize: '12px' }}>{variant.id}</td>
                    <td>{variant.title}</td>
                    <td>${variant.price.amount}</td>
                    <td>{variant.availableForSale ? 'True' : 'False'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>

          {!isParity && (
            <section id="methods">
              <h2 className="section-title">Methods</h2>

              <div className="method-block">
                <div className="method-signature">
                  <span className="kw">def</span> <span className="func">session.cart.add</span>(<span className="prm">variant_id</span>: <span className="str">str</span>, <span className="prm">quantity</span>: <span className="str">int</span> = 1) -&gt; Cart
                </div>

                <form action="/agent/actions/add" method="POST" data-agent-id={`action:add_to_cart:${product.handle}`}>
                  <input type="hidden" name="productHandle" value={product.handle} />

                  <div className="form-row">
                    <label htmlFor="variantId"># Select variant_id:</label>
                    <select name="variantId" id="variantId" required>
                      {product.variants.map(variant => (
                        <option key={variant.id} value={variant.id}>
                          {variant.id} ({variant.title} - ${variant.price.amount})
                        </option>
                      ))}
                    </select>
                  </div>

                  <div className="form-row">
                    <label htmlFor="quantity"># Set quantity:</label>
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

                  <button type="submit" className="execute-btn" data-agent-id={`submit:add_to_cart:${product.handle}`}>
                    &gt;&gt;&gt; session.cart.add(variant_id=&quot;...&quot;)
                  </button>
                </form>
              </div>
            </section>
          )}

          {isParity && (
            <section id="methods">
              <h2 className="section-title">Methods</h2>
              <p style={{ color: '#666', fontStyle: 'italic', padding: '15px 0' }}>
                [Parity mode: Use the regular product pages to add items to cart]
              </p>
            </section>
          )}

          <div className="doc-footer">
            <a href="/agent">&larr; Back to Module Index</a>
            <div style={{ marginTop: '10px' }}>Commerce API Reference v1.0 | Generated: {new Date().toISOString()}</div>
          </div>
        </main>
      </div>
    </>
  );
}

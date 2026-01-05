import { getCart, getProducts } from 'lib/shopify';
import { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Agent Dashboard - Agent Store',
  description: 'AI-friendly dashboard for Agent Store'
};

export default async function AgentDashboard() {
  const [cart, products] = await Promise.all([
    getCart(),
    getProducts({})
  ]);

  const cartJson = cart ? JSON.stringify(cart, null, 2) : '{ "status": "empty" }';

  return (
    <html lang="en">
      <head>
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <meta name="robots" content="noindex" />
        <title>Agent Dashboard - Agent Store</title>
        <style dangerouslySetInnerHTML={{ __html: `
          body { font-family: system-ui, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
          table { width: 100%; border-collapse: collapse; margin: 20px 0; }
          th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }
          th { background: #f5f5f5; }
          pre { background: #f5f5f5; padding: 15px; overflow-x: auto; }
          a { color: #0066cc; }
          nav { margin-bottom: 20px; padding: 10px; background: #f0f0f0; }
          nav a { margin-right: 15px; }
        `}} />
      </head>
      <body>
        <header>
          <h1>Agent Store Dashboard</h1>
          <nav>
            <a href="/agent">Dashboard</a>
            <a href="/llms.txt">LLMs.txt</a>
            <a href="/">Human UI</a>
          </nav>
        </header>

        <main>
          <section>
            <h2>Cart State</h2>
            <pre data-agent-id="state:cart">{cartJson}</pre>
          </section>

          <section>
            <h2>Products ({products.length})</h2>
            <table>
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Price</th>
                  <th>Available</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {products.map(product => (
                  <tr key={product.id} data-agent-id={`product:${product.handle}`}>
                    <td>
                      <a
                        href={`/agent/product/${product.handle}`}
                        data-agent-id={`nav:product:${product.handle}`}
                      >
                        {product.title}
                      </a>
                    </td>
                    <td>${product.priceRange.minVariantPrice.amount}</td>
                    <td>{product.availableForSale ? 'Yes' : 'No'}</td>
                    <td>
                      <a href={`/agent/product/${product.handle}`}>View</a>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>

          <section>
            <h2>Quick Reference</h2>
            <article>
              <h3>Available Actions</h3>
              <ul>
                <li><code>GET /agent/product/[slug]</code> - View product details</li>
                <li><code>POST /agent/actions/add</code> - Add to cart (variantId, quantity)</li>
              </ul>
            </article>
          </section>
        </main>

        <footer>
          <p>Agent Store - Shadow ACI Interface</p>
          <p>Generated: {new Date().toISOString()}</p>
        </footer>
      </body>
    </html>
  );
}

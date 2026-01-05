import { getProduct } from 'lib/shopify';
import { notFound } from 'next/navigation';

export default async function AgentProductPage({
  params
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const product = await getProduct(slug);

  if (!product) {
    notFound();
  }

  const defaultVariant = product.variants[0];

  return (
    <html lang="en">
      <head>
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <meta name="robots" content="noindex" />
        <title>{product.title} - Agent Store</title>
        <style dangerouslySetInnerHTML={{ __html: `
          body { font-family: system-ui, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
          table { width: 100%; border-collapse: collapse; margin: 20px 0; }
          th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }
          th { background: #f5f5f5; width: 30%; }
          pre { background: #f5f5f5; padding: 15px; overflow-x: auto; }
          a { color: #0066cc; }
          nav { margin-bottom: 20px; padding: 10px; background: #f0f0f0; }
          nav a { margin-right: 15px; }
          form { margin: 20px 0; padding: 20px; border: 1px solid #ccc; }
          label { display: block; margin: 10px 0 5px; }
          select, input { padding: 8px; margin-bottom: 10px; }
          button { padding: 10px 20px; background: #0066cc; color: white; border: none; cursor: pointer; }
          button:hover { background: #0055aa; }
          .success { color: green; padding: 10px; background: #e6ffe6; }
          .error { color: red; padding: 10px; background: #ffe6e6; }
        `}} />
      </head>
      <body>
        <header>
          <nav>
            <a href="/agent">Dashboard</a>
            <a href="/llms.txt">LLMs.txt</a>
            <a href="/">Human UI</a>
          </nav>
        </header>

        <main>
          <article data-agent-id={`product:${product.handle}`}>
            <h1>{product.title}</h1>

            <section>
              <h2>Product Details</h2>
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
                    <th>Price</th>
                    <td data-agent-id="product:price">
                      ${product.priceRange.minVariantPrice.amount} {product.priceRange.minVariantPrice.currencyCode}
                    </td>
                  </tr>
                  <tr>
                    <th>Available</th>
                    <td data-agent-id="product:available">
                      {product.availableForSale ? 'Yes' : 'No'}
                    </td>
                  </tr>
                  <tr>
                    <th>Description</th>
                    <td data-agent-id="product:description">{product.description}</td>
                  </tr>
                </tbody>
              </table>
            </section>

            <section>
              <h2>Variants</h2>
              <table>
                <thead>
                  <tr>
                    <th>Variant ID</th>
                    <th>Title</th>
                    <th>Price</th>
                    <th>Available</th>
                  </tr>
                </thead>
                <tbody>
                  {product.variants.map(variant => (
                    <tr key={variant.id} data-agent-id={`variant:${variant.id}`}>
                      <td>{variant.id}</td>
                      <td>{variant.title}</td>
                      <td>${variant.price.amount}</td>
                      <td>{variant.availableForSale ? 'Yes' : 'No'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>

            <section>
              <h2>Add to Cart</h2>
              <form action="/agent/actions/add" method="POST" data-agent-id={`action:add_to_cart:${product.handle}`}>
                <input type="hidden" name="productHandle" value={product.handle} />

                <label htmlFor="variantId">Variant:</label>
                <select name="variantId" id="variantId" required>
                  {product.variants.map(variant => (
                    <option key={variant.id} value={variant.id}>
                      {variant.title} - ${variant.price.amount}
                    </option>
                  ))}
                </select>

                <label htmlFor="quantity">Quantity:</label>
                <input
                  type="number"
                  name="quantity"
                  id="quantity"
                  defaultValue={1}
                  min={1}
                  max={10}
                  required
                />

                <button type="submit" data-agent-id={`submit:add_to_cart:${product.handle}`}>
                  Add to Cart
                </button>
              </form>
            </section>

            <section>
              <h2>Raw Data</h2>
              <pre data-agent-id="product:json">{JSON.stringify(product, null, 2)}</pre>
            </section>
          </article>
        </main>

        <footer>
          <p><a href="/agent">Back to Dashboard</a></p>
          <p>Generated: {new Date().toISOString()}</p>
        </footer>
      </body>
    </html>
  );
}

import { getCart, getProducts } from 'lib/shopify';

export const metadata = {
  title: 'Agent Store',
  description: 'Compact agent interface'
};

export default async function AgentDashboard() {
  const [cart, products] = await Promise.all([
    getCart(),
    getProducts({})
  ]);

  const cartItems = cart?.lines || [];
  const cartTotal = cart?.cost.totalAmount.amount || '0.00';

  return (
    <>
      <style dangerouslySetInnerHTML={{ __html: `
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
          font-family: Arial, sans-serif;
          font-size: 12px;
          padding: 8px;
          background: #fff;
        }
        h1 { font-size: 14px; margin-bottom: 8px; }
        table { width: 100%; border-collapse: collapse; margin-bottom: 12px; }
        th, td { border: 1px solid #999; padding: 4px 6px; text-align: left; }
        th { background: #e0e0e0; font-weight: bold; }
        input[type="number"] { width: 40px; padding: 2px; }
        select { padding: 2px; max-width: 80px; }
        input[type="text"], input[type="email"] { padding: 2px 4px; width: 120px; }
        button { padding: 3px 8px; cursor: pointer; background: #333; color: #fff; border: none; }
        button:hover { background: #555; }
        .cart-section { background: #f5f5f5; padding: 8px; margin-bottom: 12px; border: 1px solid #999; }
        .cart-row { display: flex; gap: 8px; align-items: center; margin-bottom: 4px; }
        .checkout-row { display: flex; gap: 8px; align-items: center; margin-top: 8px; }
        a { color: #0066cc; }
        .empty { color: #666; font-style: italic; }
      `}} />

      <h1>AGENT STORE</h1>

      {/* CART - Compact summary */}
      <div className="cart-section">
        <strong>CART</strong> ({cartItems.length} items, ${cartTotal})
        {cartItems.length === 0 ? (
          <span className="empty"> - Empty</span>
        ) : (
          <table style={{ marginTop: '4px' }}>
            <thead>
              <tr>
                <th>Product</th>
                <th>Variant</th>
                <th>Qty</th>
                <th>Price</th>
              </tr>
            </thead>
            <tbody>
              {cartItems.map(item => (
                <tr key={item.id}>
                  <td>{item.merchandise.product.handle}</td>
                  <td>{item.merchandise.title}</td>
                  <td>{item.quantity}</td>
                  <td>${item.cost.totalAmount.amount}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {cartItems.length > 0 && (
          <form action="/checkout/complete" method="POST" className="checkout-row">
            <label>Name:</label>
            <input type="text" name="name" required placeholder="Your Name" />
            <label>Email:</label>
            <input type="email" name="email" required placeholder="you@email.com" />
            <button type="submit">CHECKOUT (${cartTotal})</button>
          </form>
        )}
      </div>

      {/* PRODUCTS - All in one table with inline add-to-cart */}
      <table>
        <thead>
          <tr>
            <th>Product</th>
            <th>Price</th>
            <th>Variant</th>
            <th>Qty</th>
            <th>Action</th>
          </tr>
        </thead>
        <tbody>
          {products.map(product => (
            <tr key={product.id}>
              <td>{product.handle}</td>
              <td>${product.priceRange.minVariantPrice.amount}</td>
              <td>
                <form action="/agent/actions/add" method="POST" style={{ display: 'contents' }}>
                  <input type="hidden" name="productHandle" value={product.handle} />
                  {product.variants.length > 1 ? (
                    <select name="variantId">
                      {product.variants.map(v => (
                        <option key={v.id} value={v.id}>{v.title}</option>
                      ))}
                    </select>
                  ) : (
                    <>
                      <span>-</span>
                      <input type="hidden" name="variantId" value={product.variants[0]?.id} />
                    </>
                  )}
              </td>
              <td>
                  <input type="number" name="quantity" defaultValue={1} min={1} max={99} />
              </td>
              <td>
                  <button type="submit">ADD</button>
                </form>
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <div style={{ fontSize: '10px', color: '#666', marginTop: '8px' }}>
        <a href="/">Human UI</a> | <a href="/agent">Refresh</a>
      </div>
    </>
  );
}

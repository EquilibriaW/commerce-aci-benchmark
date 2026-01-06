import { getCart, getProducts } from 'lib/shopify';
import { cookies } from 'next/headers';

export const metadata = {
  title: 'Agent Store',
  description: 'Compact agent interface'
};

export default async function AgentDashboard() {
  const [cart, products, cookieStore] = await Promise.all([
    getCart(),
    getProducts({}),
    cookies()
  ]);

  // Check capability mode - in parity mode, hide interactive actions
  const capability = cookieStore.get('bench_capability')?.value;
  const isParity = capability === 'parity';

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
        th, td { border: 1px solid #999; padding: 4px 6px; text-align: left; vertical-align: middle; }
        th { background: #e0e0e0; font-weight: bold; }
        input[type="number"] { width: 40px; padding: 2px; }
        select { padding: 2px; max-width: 80px; }
        input[type="text"], input[type="email"] { padding: 2px 4px; width: 120px; }
        button { padding: 3px 8px; cursor: pointer; background: #333; color: #fff; border: none; }
        button:hover { background: #555; }
        .cart-section { background: #f5f5f5; padding: 8px; margin-bottom: 12px; border: 1px solid #999; }
        .checkout-row { display: flex; gap: 8px; align-items: center; margin-top: 8px; flex-wrap: wrap; }
        a { color: #0066cc; }
        .empty { color: #666; font-style: italic; }
        .inline-form { display: inline-flex; gap: 4px; align-items: center; }
      `}} />

      <h1>AGENT STORE</h1>

      {/* CART - Compact summary with edit controls */}
      <div className="cart-section">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span><strong>CART</strong> ({cartItems.length} items, ${cartTotal})</span>
          {!isParity && cartItems.length > 0 && (
            <form action="/agent/actions/clear" method="POST" style={{ display: 'inline' }}>
              <button type="submit" style={{ background: '#c00', fontSize: '10px', padding: '2px 6px' }}>CLEAR CART</button>
            </form>
          )}
        </div>
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
                {!isParity && <th>Actions</th>}
              </tr>
            </thead>
            <tbody>
              {cartItems.map(item => (
                <tr key={item.id}>
                  <td>{item.merchandise.product.handle}</td>
                  <td>{item.merchandise.title}</td>
                  <td>
                    {isParity ? (
                      item.quantity
                    ) : (
                      <form action="/agent/actions/update" method="POST" className="inline-form">
                        <input type="hidden" name="lineId" value={item.id || ''} />
                        <input type="hidden" name="merchandiseId" value={item.merchandise.id} />
                        <input type="number" name="quantity" defaultValue={item.quantity} min={1} max={99} style={{ width: '35px' }} />
                        <button type="submit" style={{ fontSize: '10px', padding: '1px 4px' }}>UPDATE</button>
                      </form>
                    )}
                  </td>
                  <td>${item.cost.totalAmount.amount}</td>
                  {!isParity && (
                    <td>
                      <form action="/agent/actions/remove" method="POST" style={{ display: 'inline' }}>
                        <input type="hidden" name="lineId" value={item.id || ''} />
                        <button type="submit" style={{ background: '#900', fontSize: '10px', padding: '1px 4px' }}>REMOVE</button>
                      </form>
                    </td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {!isParity && cartItems.length > 0 && (
          <form action="/checkout/complete" method="POST" className="checkout-row">
            <label>Name:</label>
            <input type="text" name="name" required placeholder="Your Name" />
            <label>Email:</label>
            <input type="email" name="email" required placeholder="you@email.com" />
            <button type="submit">CHECKOUT (${cartTotal})</button>
          </form>
        )}
        {isParity && cartItems.length > 0 && (
          <p style={{ marginTop: '8px', color: '#666', fontStyle: 'italic' }}>
            [Parity mode: Use the regular product pages to modify cart and checkout]
          </p>
        )}
      </div>

      {/* PRODUCTS - All in one table with inline add-to-cart */}
      <table>
        <thead>
          <tr>
            <th>Product</th>
            <th>Price</th>
            {!isParity && <th>Add to Cart</th>}
          </tr>
        </thead>
        <tbody>
          {products.map(product => (
            <tr key={product.id}>
              <td>
                <a href={`/product/${product.handle}`}>{product.handle}</a>
              </td>
              <td>${product.priceRange.minVariantPrice.amount}</td>
              {!isParity && (
                <td>
                  <form action="/agent/actions/add" method="POST" className="inline-form">
                    <input type="hidden" name="productHandle" value={product.handle} />
                    {product.variants.length > 1 ? (
                      <select name="variantId">
                        {product.variants.map(v => (
                          <option key={v.id} value={v.id}>{v.title}</option>
                        ))}
                      </select>
                    ) : (
                      <input type="hidden" name="variantId" value={product.variants[0]?.id} />
                    )}
                    <input type="number" name="quantity" defaultValue={1} min={1} max={99} />
                    <button type="submit">ADD</button>
                  </form>
                </td>
              )}
            </tr>
          ))}
        </tbody>
      </table>
      {isParity && (
        <p style={{ fontSize: '10px', color: '#666', fontStyle: 'italic', marginTop: '4px' }}>
          [Parity mode: Click product name to view details and add to cart]
        </p>
      )}

      <div style={{ fontSize: '10px', color: '#666', marginTop: '8px' }}>
        <a href="/">Human UI</a> | <a href="/agent">Refresh</a>
      </div>
    </>
  );
}

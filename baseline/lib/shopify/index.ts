// Baseline: Using mock provider (no Shopify API keys required)

export {
  addToCart,
  clearAllCarts,
  createCart,
  getCart,
  getCollection,
  getCollectionProducts,
  getCollections,
  getMenu,
  getPage,
  getPages,
  getProduct,
  getProductRecommendations,
  getProducts,
  removeFromCart,
  revalidate,
  updateCart
} from '../mock';

// Log provider
if (typeof window === 'undefined') {
  console.log('[Commerce Baseline] Using MOCK provider');
}

// Provider switcher: Uses mock data when SHOPIFY_STORE_DOMAIN is not set

const useMock = !process.env.SHOPIFY_STORE_DOMAIN;

// Re-export everything from the appropriate provider
export {
  addToCart,
  clearAllCarts,
  clearAllOrders,
  createCart,
  deleteCompletedOrder,
  deleteStoredCart,
  getCart,
  getCollection,
  getCollectionProducts,
  getCollections,
  getCompletedOrder,
  getLastCompletedOrder,
  getMenu,
  getPage,
  getPages,
  getProduct,
  getProductRecommendations,
  getProducts,
  removeFromCart,
  revalidate,
  saveCompletedOrder,
  setStoredCart,
  updateCart
} from '../mock';

export type { CompletedOrder } from '../mock';

// Log which provider is being used
if (typeof window === 'undefined') {
  console.log(`[Commerce] Using ${useMock ? 'MOCK' : 'SHOPIFY'} provider`);
}

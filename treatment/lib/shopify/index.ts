// Provider switcher: Uses mock data when SHOPIFY_STORE_DOMAIN is not set

const useMock = !process.env.SHOPIFY_STORE_DOMAIN;

// Re-export everything from the appropriate provider
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

// Log which provider is being used
if (typeof window === 'undefined') {
  console.log(`[Commerce] Using ${useMock ? 'MOCK' : 'SHOPIFY'} provider`);
}

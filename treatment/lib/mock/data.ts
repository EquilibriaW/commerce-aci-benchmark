import { Product, Collection, Cart, Menu, Page } from '../shopify/types';

// Mock product data
export const mockProducts: Product[] = [
  {
    id: 'gid://mock/Product/1',
    handle: 'classic-leather-jacket',
    availableForSale: true,
    title: 'Classic Leather Jacket',
    description: 'A timeless leather jacket crafted from premium materials.',
    descriptionHtml: '<p>A timeless leather jacket crafted from premium materials.</p>',
    options: [{ id: 'opt-1', name: 'Size', values: ['S', 'M', 'L', 'XL'] }],
    priceRange: {
      maxVariantPrice: { amount: '299.00', currencyCode: 'USD' },
      minVariantPrice: { amount: '299.00', currencyCode: 'USD' }
    },
    variants: [
      { id: 'var-1-s', title: 'S', availableForSale: true, selectedOptions: [{ name: 'Size', value: 'S' }], price: { amount: '299.00', currencyCode: 'USD' } },
      { id: 'var-1-m', title: 'M', availableForSale: true, selectedOptions: [{ name: 'Size', value: 'M' }], price: { amount: '299.00', currencyCode: 'USD' } },
      { id: 'var-1-l', title: 'L', availableForSale: true, selectedOptions: [{ name: 'Size', value: 'L' }], price: { amount: '299.00', currencyCode: 'USD' } },
      { id: 'var-1-xl', title: 'XL', availableForSale: true, selectedOptions: [{ name: 'Size', value: 'XL' }], price: { amount: '299.00', currencyCode: 'USD' } }
    ],
    featuredImage: { url: 'https://placehold.co/600x800/1a1a1a/white?text=Leather+Jacket', altText: 'Classic Leather Jacket', width: 600, height: 800 },
    images: [{ url: 'https://placehold.co/600x800/1a1a1a/white?text=Leather+Jacket', altText: 'Classic Leather Jacket', width: 600, height: 800 }],
    seo: { title: 'Classic Leather Jacket', description: 'Premium leather jacket' },
    tags: ['jacket', 'leather', 'outerwear'],
    updatedAt: new Date().toISOString()
  },
  {
    id: 'gid://mock/Product/2',
    handle: 'minimalist-watch',
    availableForSale: true,
    title: 'Minimalist Watch',
    description: 'Clean, elegant timepiece with Swiss movement.',
    descriptionHtml: '<p>Clean, elegant timepiece with Swiss movement.</p>',
    options: [{ id: 'opt-2', name: 'Color', values: ['Black', 'Silver', 'Gold'] }],
    priceRange: {
      maxVariantPrice: { amount: '199.00', currencyCode: 'USD' },
      minVariantPrice: { amount: '199.00', currencyCode: 'USD' }
    },
    variants: [
      { id: 'var-2-black', title: 'Black', availableForSale: true, selectedOptions: [{ name: 'Color', value: 'Black' }], price: { amount: '199.00', currencyCode: 'USD' } },
      { id: 'var-2-silver', title: 'Silver', availableForSale: true, selectedOptions: [{ name: 'Color', value: 'Silver' }], price: { amount: '199.00', currencyCode: 'USD' } },
      { id: 'var-2-gold', title: 'Gold', availableForSale: true, selectedOptions: [{ name: 'Color', value: 'Gold' }], price: { amount: '199.00', currencyCode: 'USD' } }
    ],
    featuredImage: { url: 'https://placehold.co/600x800/2d2d2d/white?text=Watch', altText: 'Minimalist Watch', width: 600, height: 800 },
    images: [{ url: 'https://placehold.co/600x800/2d2d2d/white?text=Watch', altText: 'Minimalist Watch', width: 600, height: 800 }],
    seo: { title: 'Minimalist Watch', description: 'Elegant Swiss watch' },
    tags: ['watch', 'accessories', 'jewelry'],
    updatedAt: new Date().toISOString()
  },
  {
    id: 'gid://mock/Product/3',
    handle: 'organic-cotton-tee',
    availableForSale: true,
    title: 'Organic Cotton T-Shirt',
    description: 'Soft, sustainable cotton tee in classic colors.',
    descriptionHtml: '<p>Soft, sustainable cotton tee in classic colors.</p>',
    options: [
      { id: 'opt-3a', name: 'Size', values: ['XS', 'S', 'M', 'L', 'XL'] },
      { id: 'opt-3b', name: 'Color', values: ['White', 'Black', 'Navy'] }
    ],
    priceRange: {
      maxVariantPrice: { amount: '45.00', currencyCode: 'USD' },
      minVariantPrice: { amount: '45.00', currencyCode: 'USD' }
    },
    variants: [
      { id: 'var-3-s-white', title: 'S / White', availableForSale: true, selectedOptions: [{ name: 'Size', value: 'S' }, { name: 'Color', value: 'White' }], price: { amount: '45.00', currencyCode: 'USD' } },
      { id: 'var-3-m-white', title: 'M / White', availableForSale: true, selectedOptions: [{ name: 'Size', value: 'M' }, { name: 'Color', value: 'White' }], price: { amount: '45.00', currencyCode: 'USD' } },
      { id: 'var-3-l-black', title: 'L / Black', availableForSale: true, selectedOptions: [{ name: 'Size', value: 'L' }, { name: 'Color', value: 'Black' }], price: { amount: '45.00', currencyCode: 'USD' } }
    ],
    featuredImage: { url: 'https://placehold.co/600x800/f5f5f5/333?text=Cotton+Tee', altText: 'Organic Cotton T-Shirt', width: 600, height: 800 },
    images: [{ url: 'https://placehold.co/600x800/f5f5f5/333?text=Cotton+Tee', altText: 'Organic Cotton T-Shirt', width: 600, height: 800 }],
    seo: { title: 'Organic Cotton T-Shirt', description: 'Sustainable cotton tee' },
    tags: ['tshirt', 'organic', 'basics'],
    updatedAt: new Date().toISOString()
  },
  {
    id: 'gid://mock/Product/4',
    handle: 'wool-blend-scarf',
    availableForSale: true,
    title: 'Wool Blend Scarf',
    description: 'Warm, cozy scarf perfect for winter.',
    descriptionHtml: '<p>Warm, cozy scarf perfect for winter.</p>',
    options: [{ id: 'opt-4', name: 'Color', values: ['Charcoal', 'Burgundy', 'Camel'] }],
    priceRange: {
      maxVariantPrice: { amount: '85.00', currencyCode: 'USD' },
      minVariantPrice: { amount: '85.00', currencyCode: 'USD' }
    },
    variants: [
      { id: 'var-4-charcoal', title: 'Charcoal', availableForSale: true, selectedOptions: [{ name: 'Color', value: 'Charcoal' }], price: { amount: '85.00', currencyCode: 'USD' } },
      { id: 'var-4-burgundy', title: 'Burgundy', availableForSale: true, selectedOptions: [{ name: 'Color', value: 'Burgundy' }], price: { amount: '85.00', currencyCode: 'USD' } }
    ],
    featuredImage: { url: 'https://placehold.co/600x800/4a3f35/white?text=Scarf', altText: 'Wool Blend Scarf', width: 600, height: 800 },
    images: [{ url: 'https://placehold.co/600x800/4a3f35/white?text=Scarf', altText: 'Wool Blend Scarf', width: 600, height: 800 }],
    seo: { title: 'Wool Blend Scarf', description: 'Winter wool scarf' },
    tags: ['scarf', 'accessories', 'winter'],
    updatedAt: new Date().toISOString()
  },
  {
    id: 'gid://mock/Product/5',
    handle: 'canvas-sneakers',
    availableForSale: true,
    title: 'Canvas Sneakers',
    description: 'Classic canvas sneakers with rubber sole.',
    descriptionHtml: '<p>Classic canvas sneakers with rubber sole.</p>',
    options: [
      { id: 'opt-5a', name: 'Size', values: ['7', '8', '9', '10', '11', '12'] },
      { id: 'opt-5b', name: 'Color', values: ['White', 'Black'] }
    ],
    priceRange: {
      maxVariantPrice: { amount: '79.00', currencyCode: 'USD' },
      minVariantPrice: { amount: '79.00', currencyCode: 'USD' }
    },
    variants: [
      { id: 'var-5-9-white', title: '9 / White', availableForSale: true, selectedOptions: [{ name: 'Size', value: '9' }, { name: 'Color', value: 'White' }], price: { amount: '79.00', currencyCode: 'USD' } },
      { id: 'var-5-10-white', title: '10 / White', availableForSale: true, selectedOptions: [{ name: 'Size', value: '10' }, { name: 'Color', value: 'White' }], price: { amount: '79.00', currencyCode: 'USD' } },
      { id: 'var-5-9-black', title: '9 / Black', availableForSale: true, selectedOptions: [{ name: 'Size', value: '9' }, { name: 'Color', value: 'Black' }], price: { amount: '79.00', currencyCode: 'USD' } }
    ],
    featuredImage: { url: 'https://placehold.co/600x800/f0f0f0/333?text=Sneakers', altText: 'Canvas Sneakers', width: 600, height: 800 },
    images: [{ url: 'https://placehold.co/600x800/f0f0f0/333?text=Sneakers', altText: 'Canvas Sneakers', width: 600, height: 800 }],
    seo: { title: 'Canvas Sneakers', description: 'Classic canvas shoes' },
    tags: ['shoes', 'sneakers', 'footwear'],
    updatedAt: new Date().toISOString()
  },
  {
    id: 'gid://mock/Product/6',
    handle: 'denim-jeans',
    availableForSale: true,
    title: 'Slim Fit Denim Jeans',
    description: 'Premium denim with modern slim fit.',
    descriptionHtml: '<p>Premium denim with modern slim fit.</p>',
    options: [
      { id: 'opt-6a', name: 'Size', values: ['28', '30', '32', '34', '36'] },
      { id: 'opt-6b', name: 'Wash', values: ['Dark', 'Medium', 'Light'] }
    ],
    priceRange: {
      maxVariantPrice: { amount: '129.00', currencyCode: 'USD' },
      minVariantPrice: { amount: '129.00', currencyCode: 'USD' }
    },
    variants: [
      { id: 'var-6-32-dark', title: '32 / Dark', availableForSale: true, selectedOptions: [{ name: 'Size', value: '32' }, { name: 'Wash', value: 'Dark' }], price: { amount: '129.00', currencyCode: 'USD' } },
      { id: 'var-6-34-medium', title: '34 / Medium', availableForSale: true, selectedOptions: [{ name: 'Size', value: '34' }, { name: 'Wash', value: 'Medium' }], price: { amount: '129.00', currencyCode: 'USD' } }
    ],
    featuredImage: { url: 'https://placehold.co/600x800/1e3a5f/white?text=Denim', altText: 'Slim Fit Denim Jeans', width: 600, height: 800 },
    images: [{ url: 'https://placehold.co/600x800/1e3a5f/white?text=Denim', altText: 'Slim Fit Denim Jeans', width: 600, height: 800 }],
    seo: { title: 'Slim Fit Denim Jeans', description: 'Premium denim jeans' },
    tags: ['jeans', 'denim', 'pants'],
    updatedAt: new Date().toISOString()
  },
  {
    id: 'gid://mock/Product/7',
    handle: 'linen-shirt',
    availableForSale: true,
    title: 'Linen Button-Down Shirt',
    description: 'Breathable linen shirt for warm weather.',
    descriptionHtml: '<p>Breathable linen shirt for warm weather.</p>',
    options: [
      { id: 'opt-7a', name: 'Size', values: ['S', 'M', 'L', 'XL'] },
      { id: 'opt-7b', name: 'Color', values: ['White', 'Sky Blue', 'Sand'] }
    ],
    priceRange: {
      maxVariantPrice: { amount: '95.00', currencyCode: 'USD' },
      minVariantPrice: { amount: '95.00', currencyCode: 'USD' }
    },
    variants: [
      { id: 'var-7-m-white', title: 'M / White', availableForSale: true, selectedOptions: [{ name: 'Size', value: 'M' }, { name: 'Color', value: 'White' }], price: { amount: '95.00', currencyCode: 'USD' } },
      { id: 'var-7-l-blue', title: 'L / Sky Blue', availableForSale: true, selectedOptions: [{ name: 'Size', value: 'L' }, { name: 'Color', value: 'Sky Blue' }], price: { amount: '95.00', currencyCode: 'USD' } }
    ],
    featuredImage: { url: 'https://placehold.co/600x800/e8e4dc/333?text=Linen+Shirt', altText: 'Linen Button-Down Shirt', width: 600, height: 800 },
    images: [{ url: 'https://placehold.co/600x800/e8e4dc/333?text=Linen+Shirt', altText: 'Linen Button-Down Shirt', width: 600, height: 800 }],
    seo: { title: 'Linen Button-Down Shirt', description: 'Breathable linen shirt' },
    tags: ['shirt', 'linen', 'summer'],
    updatedAt: new Date().toISOString()
  },
  {
    id: 'gid://mock/Product/8',
    handle: 'leather-belt',
    availableForSale: true,
    title: 'Full-Grain Leather Belt',
    description: 'Handcrafted leather belt with brass buckle.',
    descriptionHtml: '<p>Handcrafted leather belt with brass buckle.</p>',
    options: [
      { id: 'opt-8a', name: 'Size', values: ['32', '34', '36', '38', '40'] },
      { id: 'opt-8b', name: 'Color', values: ['Brown', 'Black'] }
    ],
    priceRange: {
      maxVariantPrice: { amount: '75.00', currencyCode: 'USD' },
      minVariantPrice: { amount: '75.00', currencyCode: 'USD' }
    },
    variants: [
      { id: 'var-8-34-brown', title: '34 / Brown', availableForSale: true, selectedOptions: [{ name: 'Size', value: '34' }, { name: 'Color', value: 'Brown' }], price: { amount: '75.00', currencyCode: 'USD' } },
      { id: 'var-8-36-black', title: '36 / Black', availableForSale: true, selectedOptions: [{ name: 'Size', value: '36' }, { name: 'Color', value: 'Black' }], price: { amount: '75.00', currencyCode: 'USD' } }
    ],
    featuredImage: { url: 'https://placehold.co/600x800/5c4033/white?text=Belt', altText: 'Full-Grain Leather Belt', width: 600, height: 800 },
    images: [{ url: 'https://placehold.co/600x800/5c4033/white?text=Belt', altText: 'Full-Grain Leather Belt', width: 600, height: 800 }],
    seo: { title: 'Full-Grain Leather Belt', description: 'Premium leather belt' },
    tags: ['belt', 'leather', 'accessories'],
    updatedAt: new Date().toISOString()
  },
  {
    id: 'gid://mock/Product/9',
    handle: 'cashmere-sweater',
    availableForSale: true,
    title: 'Cashmere Crew Neck Sweater',
    description: 'Luxuriously soft cashmere sweater.',
    descriptionHtml: '<p>Luxuriously soft cashmere sweater.</p>',
    options: [
      { id: 'opt-9a', name: 'Size', values: ['S', 'M', 'L', 'XL'] },
      { id: 'opt-9b', name: 'Color', values: ['Oatmeal', 'Charcoal', 'Navy'] }
    ],
    priceRange: {
      maxVariantPrice: { amount: '225.00', currencyCode: 'USD' },
      minVariantPrice: { amount: '225.00', currencyCode: 'USD' }
    },
    variants: [
      { id: 'var-9-m-oatmeal', title: 'M / Oatmeal', availableForSale: true, selectedOptions: [{ name: 'Size', value: 'M' }, { name: 'Color', value: 'Oatmeal' }], price: { amount: '225.00', currencyCode: 'USD' } },
      { id: 'var-9-l-charcoal', title: 'L / Charcoal', availableForSale: true, selectedOptions: [{ name: 'Size', value: 'L' }, { name: 'Color', value: 'Charcoal' }], price: { amount: '225.00', currencyCode: 'USD' } }
    ],
    featuredImage: { url: 'https://placehold.co/600x800/d4c8b8/333?text=Cashmere', altText: 'Cashmere Crew Neck Sweater', width: 600, height: 800 },
    images: [{ url: 'https://placehold.co/600x800/d4c8b8/333?text=Cashmere', altText: 'Cashmere Crew Neck Sweater', width: 600, height: 800 }],
    seo: { title: 'Cashmere Crew Neck Sweater', description: 'Luxury cashmere sweater' },
    tags: ['sweater', 'cashmere', 'knitwear'],
    updatedAt: new Date().toISOString()
  },
  {
    id: 'gid://mock/Product/10',
    handle: 'aviator-sunglasses',
    availableForSale: true,
    title: 'Aviator Sunglasses',
    description: 'Classic aviator sunglasses with polarized lenses.',
    descriptionHtml: '<p>Classic aviator sunglasses with polarized lenses.</p>',
    options: [{ id: 'opt-10', name: 'Frame', values: ['Gold', 'Silver', 'Black'] }],
    priceRange: {
      maxVariantPrice: { amount: '149.00', currencyCode: 'USD' },
      minVariantPrice: { amount: '149.00', currencyCode: 'USD' }
    },
    variants: [
      { id: 'var-10-gold', title: 'Gold', availableForSale: true, selectedOptions: [{ name: 'Frame', value: 'Gold' }], price: { amount: '149.00', currencyCode: 'USD' } },
      { id: 'var-10-silver', title: 'Silver', availableForSale: true, selectedOptions: [{ name: 'Frame', value: 'Silver' }], price: { amount: '149.00', currencyCode: 'USD' } }
    ],
    featuredImage: { url: 'https://placehold.co/600x800/c9a227/333?text=Aviators', altText: 'Aviator Sunglasses', width: 600, height: 800 },
    images: [{ url: 'https://placehold.co/600x800/c9a227/333?text=Aviators', altText: 'Aviator Sunglasses', width: 600, height: 800 }],
    seo: { title: 'Aviator Sunglasses', description: 'Polarized aviator shades' },
    tags: ['sunglasses', 'accessories', 'eyewear'],
    updatedAt: new Date().toISOString()
  },
  {
    id: 'gid://mock/Product/11',
    handle: 'messenger-bag',
    availableForSale: true,
    title: 'Leather Messenger Bag',
    description: 'Spacious messenger bag with laptop compartment.',
    descriptionHtml: '<p>Spacious messenger bag with laptop compartment.</p>',
    options: [{ id: 'opt-11', name: 'Color', values: ['Cognac', 'Black'] }],
    priceRange: {
      maxVariantPrice: { amount: '189.00', currencyCode: 'USD' },
      minVariantPrice: { amount: '189.00', currencyCode: 'USD' }
    },
    variants: [
      { id: 'var-11-cognac', title: 'Cognac', availableForSale: true, selectedOptions: [{ name: 'Color', value: 'Cognac' }], price: { amount: '189.00', currencyCode: 'USD' } },
      { id: 'var-11-black', title: 'Black', availableForSale: true, selectedOptions: [{ name: 'Color', value: 'Black' }], price: { amount: '189.00', currencyCode: 'USD' } }
    ],
    featuredImage: { url: 'https://placehold.co/600x800/8b5a2b/white?text=Messenger+Bag', altText: 'Leather Messenger Bag', width: 600, height: 800 },
    images: [{ url: 'https://placehold.co/600x800/8b5a2b/white?text=Messenger+Bag', altText: 'Leather Messenger Bag', width: 600, height: 800 }],
    seo: { title: 'Leather Messenger Bag', description: 'Premium messenger bag' },
    tags: ['bag', 'leather', 'accessories'],
    updatedAt: new Date().toISOString()
  },
  {
    id: 'gid://mock/Product/12',
    handle: 'wool-blazer',
    availableForSale: true,
    title: 'Wool Blend Blazer',
    description: 'Tailored wool blazer for a polished look.',
    descriptionHtml: '<p>Tailored wool blazer for a polished look.</p>',
    options: [
      { id: 'opt-12a', name: 'Size', values: ['38', '40', '42', '44', '46'] },
      { id: 'opt-12b', name: 'Color', values: ['Navy', 'Charcoal'] }
    ],
    priceRange: {
      maxVariantPrice: { amount: '349.00', currencyCode: 'USD' },
      minVariantPrice: { amount: '349.00', currencyCode: 'USD' }
    },
    variants: [
      { id: 'var-12-40-navy', title: '40 / Navy', availableForSale: true, selectedOptions: [{ name: 'Size', value: '40' }, { name: 'Color', value: 'Navy' }], price: { amount: '349.00', currencyCode: 'USD' } },
      { id: 'var-12-42-charcoal', title: '42 / Charcoal', availableForSale: true, selectedOptions: [{ name: 'Size', value: '42' }, { name: 'Color', value: 'Charcoal' }], price: { amount: '349.00', currencyCode: 'USD' } }
    ],
    featuredImage: { url: 'https://placehold.co/600x800/1a2e4a/white?text=Blazer', altText: 'Wool Blend Blazer', width: 600, height: 800 },
    images: [{ url: 'https://placehold.co/600x800/1a2e4a/white?text=Blazer', altText: 'Wool Blend Blazer', width: 600, height: 800 }],
    seo: { title: 'Wool Blend Blazer', description: 'Tailored wool blazer' },
    tags: ['blazer', 'wool', 'formal'],
    updatedAt: new Date().toISOString()
  },
  {
    id: 'gid://mock/Product/13',
    handle: 'silk-tie',
    availableForSale: true,
    title: 'Italian Silk Tie',
    description: 'Hand-finished silk tie made in Italy.',
    descriptionHtml: '<p>Hand-finished silk tie made in Italy.</p>',
    options: [{ id: 'opt-13', name: 'Pattern', values: ['Solid Navy', 'Burgundy Stripe', 'Grey Dot'] }],
    priceRange: {
      maxVariantPrice: { amount: '89.00', currencyCode: 'USD' },
      minVariantPrice: { amount: '89.00', currencyCode: 'USD' }
    },
    variants: [
      { id: 'var-13-navy', title: 'Solid Navy', availableForSale: true, selectedOptions: [{ name: 'Pattern', value: 'Solid Navy' }], price: { amount: '89.00', currencyCode: 'USD' } },
      { id: 'var-13-burgundy', title: 'Burgundy Stripe', availableForSale: true, selectedOptions: [{ name: 'Pattern', value: 'Burgundy Stripe' }], price: { amount: '89.00', currencyCode: 'USD' } }
    ],
    featuredImage: { url: 'https://placehold.co/600x800/1e3a5f/white?text=Silk+Tie', altText: 'Italian Silk Tie', width: 600, height: 800 },
    images: [{ url: 'https://placehold.co/600x800/1e3a5f/white?text=Silk+Tie', altText: 'Italian Silk Tie', width: 600, height: 800 }],
    seo: { title: 'Italian Silk Tie', description: 'Premium silk neckwear' },
    tags: ['tie', 'silk', 'formal'],
    updatedAt: new Date().toISOString()
  },
  {
    id: 'gid://mock/Product/14',
    handle: 'chino-pants',
    availableForSale: true,
    title: 'Slim Fit Chino Pants',
    description: 'Versatile cotton chinos with stretch.',
    descriptionHtml: '<p>Versatile cotton chinos with stretch.</p>',
    options: [
      { id: 'opt-14a', name: 'Size', values: ['28', '30', '32', '34', '36'] },
      { id: 'opt-14b', name: 'Color', values: ['Khaki', 'Navy', 'Olive'] }
    ],
    priceRange: {
      maxVariantPrice: { amount: '89.00', currencyCode: 'USD' },
      minVariantPrice: { amount: '89.00', currencyCode: 'USD' }
    },
    variants: [
      { id: 'var-14-32-khaki', title: '32 / Khaki', availableForSale: true, selectedOptions: [{ name: 'Size', value: '32' }, { name: 'Color', value: 'Khaki' }], price: { amount: '89.00', currencyCode: 'USD' } },
      { id: 'var-14-34-navy', title: '34 / Navy', availableForSale: true, selectedOptions: [{ name: 'Size', value: '34' }, { name: 'Color', value: 'Navy' }], price: { amount: '89.00', currencyCode: 'USD' } }
    ],
    featuredImage: { url: 'https://placehold.co/600x800/c4b29e/333?text=Chinos', altText: 'Slim Fit Chino Pants', width: 600, height: 800 },
    images: [{ url: 'https://placehold.co/600x800/c4b29e/333?text=Chinos', altText: 'Slim Fit Chino Pants', width: 600, height: 800 }],
    seo: { title: 'Slim Fit Chino Pants', description: 'Cotton chino pants' },
    tags: ['pants', 'chino', 'casual'],
    updatedAt: new Date().toISOString()
  },
  {
    id: 'gid://mock/Product/15',
    handle: 'quilted-vest',
    availableForSale: true,
    title: 'Quilted Puffer Vest',
    description: 'Lightweight quilted vest for layering.',
    descriptionHtml: '<p>Lightweight quilted vest for layering.</p>',
    options: [
      { id: 'opt-15a', name: 'Size', values: ['S', 'M', 'L', 'XL'] },
      { id: 'opt-15b', name: 'Color', values: ['Black', 'Navy', 'Olive'] }
    ],
    priceRange: {
      maxVariantPrice: { amount: '125.00', currencyCode: 'USD' },
      minVariantPrice: { amount: '125.00', currencyCode: 'USD' }
    },
    variants: [
      { id: 'var-15-m-black', title: 'M / Black', availableForSale: true, selectedOptions: [{ name: 'Size', value: 'M' }, { name: 'Color', value: 'Black' }], price: { amount: '125.00', currencyCode: 'USD' } },
      { id: 'var-15-l-navy', title: 'L / Navy', availableForSale: true, selectedOptions: [{ name: 'Size', value: 'L' }, { name: 'Color', value: 'Navy' }], price: { amount: '125.00', currencyCode: 'USD' } }
    ],
    featuredImage: { url: 'https://placehold.co/600x800/2d3436/white?text=Vest', altText: 'Quilted Puffer Vest', width: 600, height: 800 },
    images: [{ url: 'https://placehold.co/600x800/2d3436/white?text=Vest', altText: 'Quilted Puffer Vest', width: 600, height: 800 }],
    seo: { title: 'Quilted Puffer Vest', description: 'Lightweight layering vest' },
    tags: ['vest', 'outerwear', 'casual'],
    updatedAt: new Date().toISOString()
  },
  {
    id: 'gid://mock/Product/16',
    handle: 'polo-shirt',
    availableForSale: true,
    title: 'Classic Polo Shirt',
    description: 'Premium piqué polo with ribbed collar.',
    descriptionHtml: '<p>Premium piqué polo with ribbed collar.</p>',
    options: [
      { id: 'opt-16a', name: 'Size', values: ['S', 'M', 'L', 'XL', 'XXL'] },
      { id: 'opt-16b', name: 'Color', values: ['White', 'Navy', 'Forest Green'] }
    ],
    priceRange: {
      maxVariantPrice: { amount: '65.00', currencyCode: 'USD' },
      minVariantPrice: { amount: '65.00', currencyCode: 'USD' }
    },
    variants: [
      { id: 'var-16-m-white', title: 'M / White', availableForSale: true, selectedOptions: [{ name: 'Size', value: 'M' }, { name: 'Color', value: 'White' }], price: { amount: '65.00', currencyCode: 'USD' } },
      { id: 'var-16-l-navy', title: 'L / Navy', availableForSale: true, selectedOptions: [{ name: 'Size', value: 'L' }, { name: 'Color', value: 'Navy' }], price: { amount: '65.00', currencyCode: 'USD' } }
    ],
    featuredImage: { url: 'https://placehold.co/600x800/1a3d5c/white?text=Polo', altText: 'Classic Polo Shirt', width: 600, height: 800 },
    images: [{ url: 'https://placehold.co/600x800/1a3d5c/white?text=Polo', altText: 'Classic Polo Shirt', width: 600, height: 800 }],
    seo: { title: 'Classic Polo Shirt', description: 'Premium piqué polo' },
    tags: ['polo', 'shirt', 'casual'],
    updatedAt: new Date().toISOString()
  },
  {
    id: 'gid://mock/Product/17',
    handle: 'suede-loafers',
    availableForSale: true,
    title: 'Suede Penny Loafers',
    description: 'Classic penny loafers in soft suede.',
    descriptionHtml: '<p>Classic penny loafers in soft suede.</p>',
    options: [
      { id: 'opt-17a', name: 'Size', values: ['8', '9', '10', '11', '12'] },
      { id: 'opt-17b', name: 'Color', values: ['Tan', 'Navy'] }
    ],
    priceRange: {
      maxVariantPrice: { amount: '175.00', currencyCode: 'USD' },
      minVariantPrice: { amount: '175.00', currencyCode: 'USD' }
    },
    variants: [
      { id: 'var-17-10-tan', title: '10 / Tan', availableForSale: true, selectedOptions: [{ name: 'Size', value: '10' }, { name: 'Color', value: 'Tan' }], price: { amount: '175.00', currencyCode: 'USD' } },
      { id: 'var-17-11-navy', title: '11 / Navy', availableForSale: true, selectedOptions: [{ name: 'Size', value: '11' }, { name: 'Color', value: 'Navy' }], price: { amount: '175.00', currencyCode: 'USD' } }
    ],
    featuredImage: { url: 'https://placehold.co/600x800/c9a06c/333?text=Loafers', altText: 'Suede Penny Loafers', width: 600, height: 800 },
    images: [{ url: 'https://placehold.co/600x800/c9a06c/333?text=Loafers', altText: 'Suede Penny Loafers', width: 600, height: 800 }],
    seo: { title: 'Suede Penny Loafers', description: 'Classic suede loafers' },
    tags: ['shoes', 'loafers', 'footwear'],
    updatedAt: new Date().toISOString()
  },
  {
    id: 'gid://mock/Product/18',
    handle: 'merino-cardigan',
    availableForSale: true,
    title: 'Merino Wool Cardigan',
    description: 'Soft merino cardigan with button front.',
    descriptionHtml: '<p>Soft merino cardigan with button front.</p>',
    options: [
      { id: 'opt-18a', name: 'Size', values: ['S', 'M', 'L', 'XL'] },
      { id: 'opt-18b', name: 'Color', values: ['Heather Grey', 'Black', 'Burgundy'] }
    ],
    priceRange: {
      maxVariantPrice: { amount: '145.00', currencyCode: 'USD' },
      minVariantPrice: { amount: '145.00', currencyCode: 'USD' }
    },
    variants: [
      { id: 'var-18-m-grey', title: 'M / Heather Grey', availableForSale: true, selectedOptions: [{ name: 'Size', value: 'M' }, { name: 'Color', value: 'Heather Grey' }], price: { amount: '145.00', currencyCode: 'USD' } },
      { id: 'var-18-l-black', title: 'L / Black', availableForSale: true, selectedOptions: [{ name: 'Size', value: 'L' }, { name: 'Color', value: 'Black' }], price: { amount: '145.00', currencyCode: 'USD' } }
    ],
    featuredImage: { url: 'https://placehold.co/600x800/7a7a7a/white?text=Cardigan', altText: 'Merino Wool Cardigan', width: 600, height: 800 },
    images: [{ url: 'https://placehold.co/600x800/7a7a7a/white?text=Cardigan', altText: 'Merino Wool Cardigan', width: 600, height: 800 }],
    seo: { title: 'Merino Wool Cardigan', description: 'Soft merino knitwear' },
    tags: ['cardigan', 'wool', 'knitwear'],
    updatedAt: new Date().toISOString()
  },
  {
    id: 'gid://mock/Product/19',
    handle: 'canvas-tote',
    availableForSale: true,
    title: 'Canvas Tote Bag',
    description: 'Durable canvas tote with leather handles.',
    descriptionHtml: '<p>Durable canvas tote with leather handles.</p>',
    options: [{ id: 'opt-19', name: 'Color', values: ['Natural', 'Navy', 'Olive'] }],
    priceRange: {
      maxVariantPrice: { amount: '55.00', currencyCode: 'USD' },
      minVariantPrice: { amount: '55.00', currencyCode: 'USD' }
    },
    variants: [
      { id: 'var-19-natural', title: 'Natural', availableForSale: true, selectedOptions: [{ name: 'Color', value: 'Natural' }], price: { amount: '55.00', currencyCode: 'USD' } },
      { id: 'var-19-navy', title: 'Navy', availableForSale: true, selectedOptions: [{ name: 'Color', value: 'Navy' }], price: { amount: '55.00', currencyCode: 'USD' } }
    ],
    featuredImage: { url: 'https://placehold.co/600x800/e8dcc8/333?text=Tote', altText: 'Canvas Tote Bag', width: 600, height: 800 },
    images: [{ url: 'https://placehold.co/600x800/e8dcc8/333?text=Tote', altText: 'Canvas Tote Bag', width: 600, height: 800 }],
    seo: { title: 'Canvas Tote Bag', description: 'Durable canvas tote' },
    tags: ['bag', 'tote', 'accessories'],
    updatedAt: new Date().toISOString()
  },
  {
    id: 'gid://mock/Product/20',
    handle: 'rain-jacket',
    availableForSale: true,
    title: 'Waterproof Rain Jacket',
    description: 'Lightweight and packable rain protection.',
    descriptionHtml: '<p>Lightweight and packable rain protection.</p>',
    options: [
      { id: 'opt-20a', name: 'Size', values: ['S', 'M', 'L', 'XL'] },
      { id: 'opt-20b', name: 'Color', values: ['Black', 'Navy', 'Red'] }
    ],
    priceRange: {
      maxVariantPrice: { amount: '159.00', currencyCode: 'USD' },
      minVariantPrice: { amount: '159.00', currencyCode: 'USD' }
    },
    variants: [
      { id: 'var-20-m-black', title: 'M / Black', availableForSale: true, selectedOptions: [{ name: 'Size', value: 'M' }, { name: 'Color', value: 'Black' }], price: { amount: '159.00', currencyCode: 'USD' } },
      { id: 'var-20-l-navy', title: 'L / Navy', availableForSale: true, selectedOptions: [{ name: 'Size', value: 'L' }, { name: 'Color', value: 'Navy' }], price: { amount: '159.00', currencyCode: 'USD' } }
    ],
    featuredImage: { url: 'https://placehold.co/600x800/2d3e50/white?text=Rain+Jacket', altText: 'Waterproof Rain Jacket', width: 600, height: 800 },
    images: [{ url: 'https://placehold.co/600x800/2d3e50/white?text=Rain+Jacket', altText: 'Waterproof Rain Jacket', width: 600, height: 800 }],
    seo: { title: 'Waterproof Rain Jacket', description: 'Packable rain jacket' },
    tags: ['jacket', 'rain', 'outerwear'],
    updatedAt: new Date().toISOString()
  }
];

// Mock collections
export const mockCollections: Collection[] = [
  {
    handle: '',
    title: 'All',
    description: 'All products',
    seo: { title: 'All Products', description: 'Browse all products' },
    path: '/search',
    updatedAt: new Date().toISOString()
  },
  {
    handle: 'outerwear',
    title: 'Outerwear',
    description: 'Jackets, coats, and vests',
    seo: { title: 'Outerwear', description: 'Shop jackets and coats' },
    path: '/search/outerwear',
    updatedAt: new Date().toISOString()
  },
  {
    handle: 'accessories',
    title: 'Accessories',
    description: 'Belts, bags, and more',
    seo: { title: 'Accessories', description: 'Shop accessories' },
    path: '/search/accessories',
    updatedAt: new Date().toISOString()
  },
  {
    handle: 'footwear',
    title: 'Footwear',
    description: 'Shoes and sneakers',
    seo: { title: 'Footwear', description: 'Shop shoes' },
    path: '/search/footwear',
    updatedAt: new Date().toISOString()
  }
];

// Mock menu items
export const mockMenus: Record<string, Menu[]> = {
  'next-js-frontend-header-menu': [
    { title: 'All', path: '/search' },
    { title: 'Outerwear', path: '/search/outerwear' },
    { title: 'Accessories', path: '/search/accessories' },
    { title: 'Footwear', path: '/search/footwear' }
  ],
  'next-js-frontend-footer-menu': [
    { title: 'About', path: '/about' },
    { title: 'Terms', path: '/terms' },
    { title: 'Privacy', path: '/privacy' }
  ]
};

// Mock pages
export const mockPages: Page[] = [
  {
    id: 'page-1',
    title: 'About',
    handle: 'about',
    body: '<p>Welcome to Agent Store, your AI-friendly e-commerce destination.</p>',
    bodySummary: 'About Agent Store',
    seo: { title: 'About Us', description: 'Learn about Agent Store' },
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString()
  },
  {
    id: 'page-2',
    title: 'Terms of Service',
    handle: 'terms',
    body: '<p>Terms of Service for Agent Store.</p>',
    bodySummary: 'Terms of Service',
    seo: { title: 'Terms', description: 'Terms of Service' },
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString()
  },
  {
    id: 'page-3',
    title: 'Privacy Policy',
    handle: 'privacy',
    body: '<p>Privacy Policy for Agent Store.</p>',
    bodySummary: 'Privacy Policy',
    seo: { title: 'Privacy', description: 'Privacy Policy' },
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString()
  }
];

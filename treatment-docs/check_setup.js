#!/usr/bin/env node

/**
 * ACI Benchmark Verification Script
 * Tests both Treatment (ACI) and Baseline (Control) instances
 */

const TREATMENT_URL = process.env.TREATMENT_URL || 'http://localhost:3000';
const BASELINE_URL = process.env.BASELINE_URL || 'http://localhost:3001';

async function checkTreatment() {
  console.log('\n========================================');
  console.log('  TREATMENT (ACI) - Port 3000');
  console.log('========================================\n');

  let passed = 0;
  let total = 0;

  // Test 1: Homepage
  total++;
  try {
    const res = await fetch(TREATMENT_URL);
    if (res.ok) {
      console.log('  [PASS] Homepage returns 200');
      passed++;
    } else {
      console.log(`  [FAIL] Homepage returns ${res.status}`);
    }
  } catch (e) {
    console.log(`  [FAIL] Homepage error: ${e.message}`);
  }

  // Test 2: /llms.txt
  total++;
  try {
    const res = await fetch(`${TREATMENT_URL}/llms.txt`);
    const contentType = res.headers.get('content-type');
    if (res.ok && contentType?.includes('text/markdown')) {
      console.log('  [PASS] /llms.txt returns markdown');
      passed++;
    } else {
      console.log(`  [FAIL] /llms.txt: ${res.status}, type: ${contentType}`);
    }
  } catch (e) {
    console.log(`  [FAIL] /llms.txt error: ${e.message}`);
  }

  // Test 3: /agent dashboard
  total++;
  try {
    const res = await fetch(`${TREATMENT_URL}/agent`);
    const body = await res.text();
    if (res.ok && body.includes('data-agent-id')) {
      console.log('  [PASS] /agent dashboard with semantic tags');
      passed++;
    } else {
      console.log(`  [FAIL] /agent: ${res.status}`);
    }
  } catch (e) {
    console.log(`  [FAIL] /agent error: ${e.message}`);
  }

  // Test 4: /agent/product/[slug]
  total++;
  try {
    const res = await fetch(`${TREATMENT_URL}/agent/product/classic-leather-jacket`);
    const body = await res.text();
    if (res.ok && body.includes('action="/agent/actions/add"')) {
      console.log('  [PASS] /agent/product with form action');
      passed++;
    } else {
      console.log(`  [FAIL] /agent/product: ${res.status}`);
    }
  } catch (e) {
    console.log(`  [FAIL] /agent/product error: ${e.message}`);
  }

  // Test 5: Middleware - GPTBot routing
  total++;
  try {
    const res = await fetch(TREATMENT_URL, {
      headers: { 'User-Agent': 'GPTBot' }
    });
    const body = await res.text();
    if (res.ok && body.includes('Agent Dashboard')) {
      console.log('  [PASS] Middleware routes GPTBot to /agent');
      passed++;
    } else {
      console.log('  [FAIL] Middleware not routing GPTBot');
    }
  } catch (e) {
    console.log(`  [FAIL] Middleware error: ${e.message}`);
  }

  // Test 6: Middleware - ?mode=human bypass
  total++;
  try {
    const res = await fetch(`${TREATMENT_URL}/?mode=human`, {
      headers: { 'User-Agent': 'GPTBot' }
    });
    const body = await res.text();
    if (res.ok && !body.includes('Agent Dashboard')) {
      console.log('  [PASS] ?mode=human bypasses bot routing');
      passed++;
    } else {
      console.log('  [FAIL] ?mode=human not working');
    }
  } catch (e) {
    console.log(`  [FAIL] ?mode=human error: ${e.message}`);
  }

  console.log(`\n  Treatment: ${passed}/${total} tests passed`);
  return { passed, total };
}

async function checkBaseline() {
  console.log('\n========================================');
  console.log('  BASELINE (Control) - Port 3001');
  console.log('========================================\n');

  let passed = 0;
  let total = 0;

  // Test 1: Homepage
  total++;
  try {
    const res = await fetch(BASELINE_URL);
    if (res.ok) {
      console.log('  [PASS] Homepage returns 200');
      passed++;
    } else {
      console.log(`  [FAIL] Homepage returns ${res.status}`);
    }
  } catch (e) {
    console.log(`  [FAIL] Homepage error: ${e.message}`);
  }

  // Test 2: Product page works
  total++;
  try {
    const res = await fetch(`${BASELINE_URL}/product/classic-leather-jacket`);
    if (res.ok) {
      console.log('  [PASS] Product page returns 200');
      passed++;
    } else {
      console.log(`  [FAIL] Product page returns ${res.status}`);
    }
  } catch (e) {
    console.log(`  [FAIL] Product page error: ${e.message}`);
  }

  // Test 3: NO /agent route (should 404 or 500)
  total++;
  try {
    const res = await fetch(`${BASELINE_URL}/agent`);
    if (!res.ok) {
      console.log('  [PASS] /agent route does NOT exist (expected)');
      passed++;
    } else {
      console.log('  [FAIL] /agent route exists (should not)');
    }
  } catch (e) {
    console.log(`  [PASS] /agent route does NOT exist: ${e.message}`);
    passed++;
  }

  // Test 4: NO /llms.txt route
  total++;
  try {
    const res = await fetch(`${BASELINE_URL}/llms.txt`);
    if (!res.ok) {
      console.log('  [PASS] /llms.txt route does NOT exist (expected)');
      passed++;
    } else {
      console.log('  [FAIL] /llms.txt route exists (should not)');
    }
  } catch (e) {
    console.log(`  [PASS] /llms.txt route does NOT exist: ${e.message}`);
    passed++;
  }

  // Test 5: NO data-agent-id in HTML
  total++;
  try {
    const res = await fetch(`${BASELINE_URL}/search`);
    const body = await res.text();
    if (!body.includes('data-agent-id=')) {
      console.log('  [PASS] No data-agent-id attributes (expected)');
      passed++;
    } else {
      console.log('  [FAIL] Found data-agent-id attributes (should not)');
    }
  } catch (e) {
    console.log(`  [FAIL] Search page error: ${e.message}`);
  }

  console.log(`\n  Baseline: ${passed}/${total} tests passed`);
  return { passed, total };
}

async function main() {
  console.log('========================================');
  console.log('  ACI Benchmark Verification');
  console.log('========================================');

  const treatment = await checkTreatment();
  const baseline = await checkBaseline();

  const totalPassed = treatment.passed + baseline.passed;
  const totalTests = treatment.total + baseline.total;

  console.log('\n========================================');
  console.log(`  TOTAL: ${totalPassed}/${totalTests} tests passed`);
  console.log('========================================\n');

  if (totalPassed === totalTests) {
    console.log('All tests passed! Benchmark setup is ready.\n');
    console.log('Endpoints:');
    console.log(`  Treatment (ACI): ${TREATMENT_URL}`);
    console.log(`  Baseline:        ${BASELINE_URL}`);
    console.log('');
    process.exit(0);
  } else {
    console.log('Some tests failed. Please check the output above.\n');
    process.exit(1);
  }
}

main().catch(error => {
  console.error('Fatal error:', error);
  process.exit(1);
});

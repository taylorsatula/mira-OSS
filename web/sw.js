// MIRA Service Worker - Network-First Strategy
// Always fetch fresh content when online, cache as offline fallback

self.addEventListener('fetch', (event) => {
  const url = event.request.url;

  // Skip WebSocket and API requests - these shouldn't be cached
  if (url.includes('/v0/') || url.startsWith('ws://') || url.startsWith('wss://')) {
    return;
  }

  event.respondWith(
    fetch(event.request)
      .then(response => {
        // Cache successful responses for offline use
        const clone = response.clone();
        caches.open('mira-cache').then(cache => cache.put(event.request, clone));
        return response;
      })
      .catch(() => caches.match(event.request))
  );
});

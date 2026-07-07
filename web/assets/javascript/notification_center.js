/**
 * NOTIFICATION_CENTER.JS - UI for reminders and notifications
 *
 * TODO: Implement reminder bell UI
 * - Add bell button + popover to index.html
 * - Poll GET /api/tools/reminder_tool/query?operation=get_reminders&date_type=upcoming every 60s
 * - Render reminders in popover
 * - Complete button calls POST /api/actions with {domain: "reminder", action: "complete", data: {id: "..."}}
 * - Update bell glow state based on active reminder count
 *
 * Architecture: Query tool directly, NOT trinkets. Trinkets are for MIRA's context window only.
 */

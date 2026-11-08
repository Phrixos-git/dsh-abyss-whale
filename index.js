/**
 * Host half of the Abyss whale theme bundle.
 *
 * The theme itself is entirely client-side: the browser half registers an
 * `abyss-whale` theme through `ctx.theme`, adds the frame-wide deep-sea
 * atmosphere to `shell.overlay`, and contributes its on/off row to Settings →
 * General. This host row only makes the package discoverable to the client
 * module graph.
 */

/** No host-side services are needed. */
export function apply() {}

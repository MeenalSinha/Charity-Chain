# CharityChain — Phase 12 Design System & Component Library
**Document Version:** 1.0.0  
**Phase:** Phase 12 (Modern Frontend Migration)  
**Date:** November 13, 2026  

---

## 1. Design Direction & Visual Identity

CharityChain adopts a **modern fintech and charitable accountability aesthetic**:
- Professional, trust-focused blue brand palette (`#2563EB`)
- High-contrast impact green accents (`#10B981`) for verified milestones and balanced ledgers
- Clean typography using standard system font stacks (San Francisco, Segoe UI, Roboto)
- Subtle shadows, rounded cards, and clean borders
- Clear visual hierarchy with zero decorative clutter

---

## 2. Design Tokens

### Color Palette

| Token | Hex Value | Semantic Role |
|---|---|---|
| `primary` | `#2563EB` | Trust blue, primary buttons, brand accents |
| `primary-hover` | `#1D4ED8` | Primary button hover state |
| `success` | `#10B981` | Verification confirmed, funds captured, balanced ledger |
| `warning` | `#F59E0B` | Under review, pending submission |
| `danger` | `#EF4444` | Rejection, validation error, ledger imbalance |
| `background` | `#FFFFFF` | Page surface background |
| `surface-muted`| `#F9FAFB` | Card surfaces, table headers, form inputs |
| `text-primary`| `#111827` | Headings, emphasized labels |
| `text-muted`  | `#6B7280` | Subtitles, helper text, timestamps |

---

## 3. Component Catalog

All components reside in `frontend/components/ui/`:

1. **`Button`**: Accessible button with variants (`default`, `outline`, `ghost`, `destructive`, `success`, `link`) and sizes (`sm`, `md`, `lg`, `icon`).
2. **`Badge`**: Pill badge supporting semantic color coding (`default`, `success`, `warning`, `danger`, `outline`, `secondary`).
3. **`StatusBadge`**: Domain-aware status mapper translating backend enums (`ACTIVE`, `VERIFIED`, `COMPLETED`, `PENDING`, `REJECTED`, etc.) into standardized styled badges.
4. **`Card`, `CardHeader`, `CardTitle`, `CardContent`, `CardFooter`**: Modular surface containers.
5. **`KPICard`**: Telemetry and metrics display with title, value, subtitle, trend indicator, and icon slot.
6. **`ProgressBar`**: Accessible progress bar (`role="progressbar"`) with ARIA attributes and color themes.
7. **`EmptyState`**: Standardized zero-data presentation with icon, title, description, and call-to-action button.
8. **`Skeleton`, `SkeletonCard`, `SkeletonTable`**: Accessible loading skeletons matching destination layouts.
9. **`PublicHeader` & `PublicFooter`**: Responsive public navigation with mobile menu drawer.
10. **`DashboardLayout`**: Role-aware sidebar layout with collapsible mobile drawer and user profile status.


ESTIMATION_EXAMPLES : list[dict] = [
    {
        "meeting_summary": "Client needs a web platform for inventory management with real-time stock tracking, user roles, and reporting capabilities.",
        "estimation": """
        ## Estimation: Inventory Management Platform

        ### Task Breakdown:
        | Task | Hours | Cost (EUR) |
        |------|-------|------------|
        | UI/UX Design | 40 | 2500.00 |
        | Backend API (CRUD inventory) | 60 | 3750.00 |
        | Authentication and Role-Based Access | 20 | 1250.00 |
        | Dashboard with Analytics | 30 | 1875.00 |
        | Testing and QA | 25 | 1562.50 |

        **Total Estimated Hours: 175 hours**
        **Total Estimated Cost: 10937.50 EUR**
        **Recommended Team: 2 full-stack developers + 1 UX designer (part-time)**
        **Estimated Duration: 6-8 weeks**
        """
    },
    {
        "meeting_summary": "Client wants a mobile e-commerce application with product catalog, shopping cart, payment integration, and order tracking for iOS and Android.",
        "estimation": """
        ## Estimation: Mobile E-commerce Application

        ### Task Breakdown:
        | Task | Hours | Cost (EUR) |
        |------|-------|------------|
        | Mobile App UI/UX Design (iOS & Android) | 80 | 5000.00 |
        | Product Catalog & Search | 50 | 3125.00 |
        | Shopping Cart & Checkout Flow | 45 | 2812.50 |
        | Payment Gateway Integration (Stripe/PayPal) | 40 | 2500.00 |
        | Order Management System | 35 | 2187.50 |
        | User Authentication & Profile | 25 | 1562.50 |
        | Testing, Deployment & App Store Setup | 45 | 2812.50 |

        **Total Estimated Hours: 320 hours**
        **Total Estimated Cost: 20000.00 EUR**
        **Recommended Team: 2 mobile developers (1 iOS, 1 Android) + 1 backend developer + 1 QA engineer**
        **Estimated Duration: 10-12 weeks**
        """
    },
    {
        "meeting_summary": "Client requires a real-time analytics dashboard to visualize sales data, customer insights, and performance metrics with interactive charts.",
        "estimation": """
        ## Estimation: Real-time Analytics Dashboard

        ### Task Breakdown:
        | Task | Hours | Cost (EUR) |
        |------|-------|------------|
        | Dashboard UI/UX Design | 35 | 2187.50 |
        | Frontend Development (React/Vue) | 55 | 3437.50 |
        | Data Integration & APIs | 40 | 2500.00 |
        | Real-time Data Processing | 50 | 3125.00 |
        | Chart & Visualization Components | 30 | 1875.00 |
        | Database Optimization | 25 | 1562.50 |
        | Testing & Performance Tuning | 20 | 1250.00 |

        **Total Estimated Hours: 255 hours**
        **Total Estimated Cost: 15937.50 EUR**
        **Recommended Team: 1 frontend developer + 1 backend developer + 1 data engineer**
        **Estimated Duration: 8-10 weeks**
        """
    },
    {
        "meeting_summary": "Client needs a comprehensive CRM system to manage customer relationships, sales pipeline, leads, and customer communications.",
        "estimation": """
        ## Estimation: CRM System

        ### Task Breakdown:
        | Task | Hours | Cost (EUR) |
        |------|-------|------------|
        | System Architecture & Database Design | 30 | 1875.00 |
        | UI/UX Design | 50 | 3125.00 |
        | Customer Management Module | 60 | 3750.00 |
        | Sales Pipeline & Leads Management | 70 | 4375.00 |
        | Communication Tools (Email, Chat) | 50 | 3125.00 |
        | Reporting & Analytics | 40 | 2500.00 |
        | User Authentication & Permissions | 30 | 1875.00 |
        | Integration with Third-party Services | 35 | 2187.50 |
        | Testing & QA | 45 | 2812.50 |

        **Total Estimated Hours: 410 hours**
        **Total Estimated Cost: 25625.00 EUR**
        **Recommended Team: 3 full-stack developers + 1 database architect + 1 UI/UX designer + 1 QA engineer**
        **Estimated Duration: 12-14 weeks**
        """
    },
    {
        "meeting_summary": "Client wants a custom Content Management System (CMS) for managing blog posts, multimedia content, user permissions, and content scheduling.",
        "estimation": """
        ## Estimation: Content Management System (CMS)

        ### Task Breakdown:
        | Task | Hours | Cost (EUR) |
        |------|-------|------------|
        | CMS Architecture & Design | 35 | 2187.50 |
        | Admin Dashboard Development | 55 | 3437.50 |
        | Content Editor with Rich Text Support | 50 | 3125.00 |
        | User Management & Permissions | 30 | 1875.00 |
        | Media Library & File Management | 35 | 2187.50 |
        | Content Scheduling System | 25 | 1562.50 |
        | SEO Optimization Tools | 20 | 1250.00 |
        | Frontend Template System | 40 | 2500.00 |
        | Testing & Documentation | 30 | 1875.00 |

        **Total Estimated Hours: 320 hours**
        **Total Estimated Cost: 20000.00 EUR**
        **Recommended Team: 2 full-stack developers + 1 frontend developer + 1 UX designer + 1 QA engineer**
        **Estimated Duration: 10-12 weeks**
        """
    },
    {
        "meeting_summary": "Client needs an API integration service to connect multiple third-party platforms, synchronize data, and handle webhook events with error handling and logging.",
        "estimation": """
        ## Estimation: API Integration Service

        ### Task Breakdown:
        | Task | Hours | Cost (EUR) |
        |------|-------|------------|
        | System Architecture & Planning | 25 | 1562.50 |
        | Third-party API Integration (3-4 services) | 80 | 5000.00 |
        | Data Synchronization Engine | 50 | 3125.00 |
        | Webhook Handler Development | 35 | 2187.50 |
        | Error Handling & Retry Logic | 30 | 1875.00 |
        | Logging & Monitoring System | 25 | 1562.50 |
        | Rate Limiting & API Throttling | 20 | 1250.00 |
        | Documentation & API Specs | 20 | 1250.00 |
        | Testing & Performance Testing | 35 | 2187.50 |

        **Total Estimated Hours: 320 hours**
        **Total Estimated Cost: 20000.00 EUR**
        **Recommended Team: 2 backend developers + 1 DevOps engineer + 1 QA engineer**
        **Estimated Duration: 10-12 weeks**
        """
    },
]


def format_examples(examples: list[dict]) -> str:
    """Format estimation examples (list of examples) into a single string suitable for injection into a system prompt."""
    parts: list[str] = []
    for i, example in enumerate(examples, start=1):
        parts.append(
            f"--- EXAMPLE {i} ---\n"
            f"Meeting Summary:\n{example['meeting_summary']}\n\n"
            f"Estimation:\n{example['estimation']}\n"
        )
    return "\n".join(parts)

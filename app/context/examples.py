ESTIMATION_EXAMPLES : list[dict] = [
    {
        "meeting_summary": "Client needs a web platform for inventory management with real-time stock tracking, user roles, and reporting capabilities.",
        "estimation": """
        ## Estimation: Inventory Management Platform

        ### Task Breakdown:
        1. UI/UX Design: 40 hours
        2. Backend API (CRUD inventory): 60 hours
        3. Authentication and Role-Based Access: 20 hours
        4. Dashboard with Analytics: 30 hours
        5. Testing and QA: 25 hours

        **Total Estimated Hours: 175 hours**
        **Recommended Team: 2 full-stack developers + 1 UX designer (part-time)**
        **Estimated Duration: 6-8 weeks**
        """
    },
    {
        "meeting_summary": "Client wants a mobile e-commerce application with product catalog, shopping cart, payment integration, and order tracking for iOS and Android.",
        "estimation": """
        ## Estimation: Mobile E-commerce Application

        ### Task Breakdown:
        1. Mobile App UI/UX Design (iOS & Android): 80 hours
        2. Product Catalog & Search: 50 hours
        3. Shopping Cart & Checkout Flow: 45 hours
        4. Payment Gateway Integration (Stripe/PayPal): 40 hours
        5. Order Management System: 35 hours
        6. User Authentication & Profile: 25 hours
        7. Testing, Deployment & App Store Setup: 45 hours

        **Total Estimated Hours: 320 hours**
        **Recommended Team: 2 mobile developers (1 iOS, 1 Android) + 1 backend developer + 1 QA engineer**
        **Estimated Duration: 10-12 weeks**
        """
    },
    {
        "meeting_summary": "Client requires a real-time analytics dashboard to visualize sales data, customer insights, and performance metrics with interactive charts.",
        "estimation": """
        ## Estimation: Real-time Analytics Dashboard

        ### Task Breakdown:
        1. Dashboard UI/UX Design: 35 hours
        2. Frontend Development (React/Vue): 55 hours
        3. Data Integration & APIs: 40 hours
        4. Real-time Data Processing: 50 hours
        5. Chart & Visualization Components: 30 hours
        6. Database Optimization: 25 hours
        7. Testing & Performance Tuning: 20 hours

        **Total Estimated Hours: 255 hours**
        **Recommended Team: 1 frontend developer + 1 backend developer + 1 data engineer**
        **Estimated Duration: 8-10 weeks**
        """
    },
    {
        "meeting_summary": "Client needs a comprehensive CRM system to manage customer relationships, sales pipeline, leads, and customer communications.",
        "estimation": """
        ## Estimation: CRM System

        ### Task Breakdown:
        1. System Architecture & Database Design: 30 hours
        2. UI/UX Design: 50 hours
        3. Customer Management Module: 60 hours
        4. Sales Pipeline & Leads Management: 70 hours
        5. Communication Tools (Email, Chat): 50 hours
        6. Reporting & Analytics: 40 hours
        7. User Authentication & Permissions: 30 hours
        8. Integration with Third-party Services: 35 hours
        9. Testing & QA: 45 hours

        **Total Estimated Hours: 410 hours**
        **Recommended Team: 3 full-stack developers + 1 database architect + 1 UI/UX designer + 1 QA engineer**
        **Estimated Duration: 12-14 weeks**
        """
    },
    {
        "meeting_summary": "Client wants a custom Content Management System (CMS) for managing blog posts, multimedia content, user permissions, and content scheduling.",
        "estimation": """
        ## Estimation: Content Management System (CMS)

        ### Task Breakdown:
        1. CMS Architecture & Design: 35 hours
        2. Admin Dashboard Development: 55 hours
        3. Content Editor with Rich Text Support: 50 hours
        4. User Management & Permissions: 30 hours
        5. Media Library & File Management: 35 hours
        6. Content Scheduling System: 25 hours
        7. SEO Optimization Tools: 20 hours
        8. Frontend Template System: 40 hours
        9. Testing & Documentation: 30 hours

        **Total Estimated Hours: 320 hours**
        **Recommended Team: 2 full-stack developers + 1 frontend developer + 1 UX designer + 1 QA engineer**
        **Estimated Duration: 10-12 weeks**
        """
    },
    {
        "meeting_summary": "Client needs an API integration service to connect multiple third-party platforms, synchronize data, and handle webhook events with error handling and logging.",
        "estimation": """
        ## Estimation: API Integration Service

        ### Task Breakdown:
        1. System Architecture & Planning: 25 hours
        2. Third-party API Integration (3-4 services): 80 hours
        3. Data Synchronization Engine: 50 hours
        4. Webhook Handler Development: 35 hours
        5. Error Handling & Retry Logic: 30 hours
        6. Logging & Monitoring System: 25 hours
        7. Rate Limiting & API Throttling: 20 hours
        8. Documentation & API Specs: 20 hours
        9. Testing & Performance Testing: 35 hours

        **Total Estimated Hours: 320 hours**
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